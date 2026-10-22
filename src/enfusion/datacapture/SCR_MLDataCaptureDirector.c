// ============================================================================
// SCR_MLDataCaptureDirector - compiled driving director for the capture world
// ============================================================================
//
// Replaces the inline userScript of GameMode_DataCapture (worlds/DataCapture/DC_Arland_Layers/default.layer) and the superseded
// SCR_AIDrivingSimulator. Attach it to the GameMode_DataCapture entity (an SCR_BaseGameMode) next to
// SCR_MLDataCollector. It never captures anything itself: the collector owns the capture clock.
//
// WHAT IT DOES (authority only, play mode only)
//   - sets the initial time of day and weather;
//   - for every configured group id k = 1..m_iMaxAIGroups: spawns the empty group m_sGroupPrefab at
//     <m_sSpawnPointPrefix>k, hands it the placed vehicle <m_sVehiclePrefix>k and gives it its own forced-move
//     waypoints on every <m_sDriveTargetPrefix>N marker; each waypoint keeps its original marker id N;
//   - seats the group's single driver (m_sDriverPrefab) straight into the vehicle's pilot seat with the vanilla Game
//     Master "crewed" call (BaseCompartmentSlot.SpawnCharacterInCompartment: spawn, activate AI, teleport into the
//     seat, join the group), retrying up to m_iSeatAttempts times; only if every attempt fails it falls back to
//     walk-in boarding (driver spawned on foot at the spawn point, the group AI walks it to its car). The first
//     waypoint is assigned only once the driver exists, so a seated group drives straight away;
//   - keeps every capture group simulated wherever the camera is (1.8 deactivates AI at max LOD, about 1 km or more
//     from every observer): pins its agents to full AI LOD (AIAgent.SetPermanentLOD(0), group PreventMaxLOD, CRITICAL
//     spawn importance; m_bPinDriverLOD) and puts a temporary local observer on its vehicle
//     (ObserversSystem.InsertObserverSP; m_bVehicleObservers). Both are undone when the director stops. The seated,
//     statistics and boarding-window lines log the agent count, dormancy, observer range and per-agent LOD;
//   - after each completed waypoint assigns a different one, never the current one and never one within 2x the
//     completion radius of the vehicle or of the current target; when no target qualifies the group gets none and
//     the choice is retried (no fallback to an excluded target);
//   - once every group has been attempted ("Spawned N/M groups"), opens a boarding window of m_iBoardingWindowMs and
//     polls the pilot seats. It starts the SCR_MLDataCollector session as soon as every configured group has a living
//     seated driver, otherwise when the window ends, with the groups seated by then if there are at least
//     m_iMinSeatedGroups of them (default 1; m_iMinSeatedGroups = m_iMaxAIGroups is the old all-or-nothing gate).
//     Right before StartSession it registers the seated vehicles in group-id order, so vehicle_id is the registration
//     index (k - 1 when every group is seated; session_info.txt maps vehicle_<id> to AI_Car_k). Missing groups are
//     listed one per line. With no seated driver, fewer than the minimum, or a minimum that can no longer be reached
//     it logs an ERROR and starts no session. A driver seated after the session started is not added to it (the
//     collector's vehicle list and session_info.txt are fixed at StartSession): that is logged, the car keeps driving;
//   - teleports a vehicle back to its own group's spawn point when it has not moved for m_fStuckTimeS while a living
//     driver sat in its pilot seat (occupancy is re-read on every monitor tick, independent of the readiness poll);
//   - stops the collector session (SCR_MLDataCollector.StopSession) when the game mode or the game ends.
//
// LIFECYCLE (vanilla 1.8.0.13 Game/GameMode/SCR_BaseGameMode.c:428-458 and :499-521): the authority's first
// EOnFrame calls StartGameMode() -> OnGameStateChanged() -> OnGameModeStart() on every SCR_BaseGameModeComponent.
// OnGameModeStart is "called on every machine", so StartDirector() also requires Replication.IsServer():
// SCR_BaseGameMode.IsMaster() is true whenever the entity has no RplComponent (SCR_BaseGameMode.c:366-370).
// FallbackStart() starts the director once m_iFallbackStartMs of world time have passed since OnPostInit without an
// OnGameModeStart; both paths go through the same idempotent StartDirector(), so at most one of them starts it.
//
// Call queue: at most one entry per director function is ever pending (one repeating spawn tick, one repeating
// monitor tick that also runs the per-group seating retries and waypoint assignments, one readiness poll, one
// fallback timer), so a single ScriptCallQueue.Remove per function in StopDirector() clears everything the director
// queued.
//
// Failure handling: a group whose spawn point, vehicle, vehicle simulation, pilot seat, group prefab, driver prefab
// or waypoint prefab is missing or invalid is skipped with an ERROR before anything is spawned for it (no shared-car
// or shared-spawn fallback). Every callback resolves its group through SCR_MLDataCaptureGroupState records keyed by
// group id, never by array position.
// ============================================================================

#ifdef ENABLE_DIAG
//! Own DiagMenu ids. The userScript re-used SCR_AIWorld's BT debug ids (Game/AI/SCR_AIWorld.c:79-92).
//! Appending ids with a modded enum follows Autotest/Game/TestFramework/SCR_AutotestDebugMenu.c:3.
modded enum SCR_DebugMenuID
{
	DEBUGUI_MLDATACAPTURE_MENU,
	DEBUGUI_MLDATACAPTURE_PANEL,
	DEBUGUI_MLDATACAPTURE_VEHICLES,
	DEBUGUI_MLDATACAPTURE_AI_INPUTS
}
#endif

//------------------------------------------------------------------------------------------------
//! How a group's driver got into its vehicle.
enum SCR_EMLDataCaptureBoarding
{
	PENDING,	//!< direct seating attempts still running
	DIRECT,		//!< spawned straight into the pilot seat (BaseCompartmentSlot.SpawnCharacterInCompartment)
	WALK_IN,	//!< fallback: spawned on foot at the spawn point, the group AI boards the car
	FAILED		//!< no driver could be spawned for this group
}

//------------------------------------------------------------------------------------------------
//! One forced-move waypoint of one group, spawned on drive-target marker <prefix><m_iTargetId>.
class SCR_MLDataCaptureWaypoint
{
	int m_iTargetId;		//!< original marker id N of AI_Drive_Target_N (1-based), never a compacted index
	vector m_vPosition;		//!< marker world position
	AIWaypoint m_Waypoint;

	//------------------------------------------------------------------------------------------------
	void SCR_MLDataCaptureWaypoint(int targetId, vector position, AIWaypoint waypoint)
	{
		m_iTargetId = targetId;
		m_vPosition = position;
		m_Waypoint = waypoint;
	}
}

//------------------------------------------------------------------------------------------------
//! Everything the director knows about one AI group, keyed by its stable group id k (AI_Spawn_k, AI_Car_k).
class SCR_MLDataCaptureGroupState
{
	int m_iGroupId;
	SCR_AIGroup m_Group;
	Vehicle m_Vehicle;
	VehicleWheeledSimulation m_Simulation;
	IEntity m_SpawnPoint;
	ref array<ref SCR_MLDataCaptureWaypoint> m_aWaypoints = {};
	int m_iCurrentWaypoint = -1;	//!< index into m_aWaypoints, -1 = none assigned yet

	IEntity m_Driver;				//!< the driver character this director spawned for the group, null = none (yet)
	SCR_EMLDataCaptureBoarding m_eBoarding = SCR_EMLDataCaptureBoarding.PENDING;
	int m_iSeatAttemptsMade;		//!< direct seating attempts made so far
	float m_fNextSeatAttemptAtMs = -1;	//!< world time of the next direct seating attempt, -1 = none pending
	bool m_bDrivingStarted;			//!< the first waypoint was assigned (only once the driver exists)

	bool m_bLODPinned;				//!< PinGroupLOD ran and ReleaseGroup has not undone it yet
	int m_iImportanceBeforePin = -1;	//!< group spawn importance before the pin, -1 = not saved
	bool m_bObserverAdded;			//!< a temporary local observer follows the vehicle (ObserversSystem.InsertObserverSP)

	bool m_bDriverSeated;			//!< a living character sits in the pilot seat (latest monitor or readiness tick)
	float m_fFirstSeatedS = -1;		//!< director time [s] when a seated driver was first seen, -1 = never
	bool m_bSeatedLogged;			//!< the "driver seated" line was logged
	bool m_bRegistered;				//!< registered with SCR_MLDataCollector
	float m_fNextAssignAtMs = -1;	//!< world time at which the next waypoint is due, -1 = none pending
	bool m_bNoTargetWarned;			//!< the "no drive target qualifies" warning was logged since the last assignment

	int m_iWaypointsCompleted;
	int m_iStuckResets;
	float m_fDistance;
	float m_fMaxSpeed;
	float m_fMaxRPM;
	float m_fLastSpeed;
	float m_fAcceleration;			//!< km/h per second, debug panel only
	vector m_vLastPosition;
	vector m_vStuckCheckPosition;
	float m_fStuckTimeS;			//!< seconds spent within the stuck distance of m_vStuckCheckPosition
	float m_fLastMonitorTimeMs;		//!< world time of the previous monitor tick, -1 = none

	//------------------------------------------------------------------------------------------------
	void SCR_MLDataCaptureGroupState(int groupId, SCR_AIGroup group, Vehicle vehicle, VehicleWheeledSimulation simulation, IEntity spawnPoint)
	{
		m_iGroupId = groupId;
		m_Group = group;
		m_Vehicle = vehicle;
		m_Simulation = simulation;
		m_SpawnPoint = spawnPoint;
		m_vLastPosition = vehicle.GetOrigin();
		m_vStuckCheckPosition = m_vLastPosition;
		m_fLastMonitorTimeMs = -1;
	}

	//------------------------------------------------------------------------------------------------
	//! \return index into m_aWaypoints of the given waypoint entity, or -1
	int FindWaypoint(AIWaypoint waypoint)
	{
		if (!waypoint)
			return -1;

		for (int i = 0, count = m_aWaypoints.Count(); i < count; i++)
		{
			if (m_aWaypoints[i].m_Waypoint == waypoint)
				return i;
		}

		return -1;
	}

	//------------------------------------------------------------------------------------------------
	//! \return original marker id of the current waypoint, or -1
	int GetCurrentTargetId()
	{
		if (m_iCurrentWaypoint < 0 || m_iCurrentWaypoint >= m_aWaypoints.Count())
			return -1;

		return m_aWaypoints[m_iCurrentWaypoint].m_iTargetId;
	}
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Seats one AI driver directly in each placed capture vehicle, drives them between drive-target markers and starts the SCR_MLDataCollector session once the drivers are seated")]
class SCR_MLDataCaptureDirectorClass : SCR_BaseGameModeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class SCR_MLDataCaptureDirector : SCR_BaseGameModeComponent
{
	// === GROUPS AND ENTITY NAMES ===
	[Attribute("5", UIWidgets.Slider, "Number of AI groups. Group k uses the entities <spawn point prefix>k and <vehicle prefix>k", "1 10 1", category: "Groups")]
	protected int m_iMaxAIGroups;

	[Attribute("AI_Spawn_", UIWidgets.EditBox, "Name prefix of the group spawn points (AI_Spawn_1, AI_Spawn_2, ...)", category: "Groups")]
	protected string m_sSpawnPointPrefix;

	[Attribute("AI_Car_", UIWidgets.EditBox, "Name prefix of the placed vehicles, one per group (AI_Car_1, AI_Car_2, ...)", category: "Groups")]
	protected string m_sVehiclePrefix;

	[Attribute("AI_Drive_Target_", UIWidgets.EditBox, "Name prefix of the drive-target markers (AI_Drive_Target_1, ...)", category: "Groups")]
	protected string m_sDriveTargetPrefix;

	[Attribute("20", UIWidgets.Slider, "Highest drive-target marker id to look for (missing ids are skipped, the others keep their id)", "1 100 1", category: "Groups")]
	protected int m_iMaxDriveTargets;

	[Attribute("{EACD97CF4A702FAE}Prefabs/Groups/BLUFOR/Group_US_Base.et", UIWidgets.ResourcePickerThumbnail, "Empty AI group prefab (no unit slots) spawned at each spawn point; the director adds one driver to it. Its faction must match the driver prefab", "et", category: "Groups")]
	protected ResourceName m_sGroupPrefab;

	[Attribute("{26A9756790131354}Prefabs/Characters/Factions/BLUFOR/US_Army/Character_US_Rifleman.et", UIWidgets.ResourcePickerThumbnail, "Character prefab put into each vehicle's driver seat (vanilla default pilot of the M151A2 and M998)", "et", category: "Groups")]
	protected ResourceName m_sDriverPrefab;

	[Attribute("{06E1B6EBD480C6E0}Prefabs/AI/Waypoints/AIWaypoint_ForcedMove.et", UIWidgets.ResourcePickerThumbnail, "Waypoint prefab spawned on each drive-target marker (must be an AIWaypoint)", "et", category: "Groups")]
	protected ResourceName m_sWaypointPrefab;

	[Attribute("2000", UIWidgets.Slider, "Delay between group spawns [ms]; group k spawns k times this after the start", "0 10000 100", category: "Groups")]
	protected int m_iSpawnStaggerMs;

	// === BOARDING ===
	[Attribute("3", UIWidgets.Slider, "Attempts to spawn the driver straight into the vehicle's pilot seat before falling back to walk-in boarding", "1 10 1", category: "Boarding")]
	protected int m_iSeatAttempts;

	[Attribute("1000", UIWidgets.Slider, "Delay between direct seating attempts [ms]", "200 10000 100", category: "Boarding")]
	protected int m_iSeatRetryMs;

	[Attribute("1", UIWidgets.CheckBox, "Keep every capture driver at full AI LOD: AIAgent.SetPermanentLOD(0) on the group's agents, PreventMaxLOD on the group and CRITICAL spawn importance (the 1.9 vanilla logistic-driver pin). Without it, AI about 1 km or more from every observer is deactivated and cannot drive", category: "Boarding")]
	protected bool m_bPinDriverLOD;

	[Attribute("1", UIWidgets.CheckBox, "Add a temporary local observer on every capture vehicle (ObserversSystem.InsertObserverSP), so cars far from the camera stay simulated; removed when the director stops", category: "Boarding")]
	protected bool m_bVehicleObservers;

	//! ObserversSystem SP key of group k's vehicle observer: OBSERVER_KEY_BASE + k (no vanilla script inserts SP observers)
	static const int OBSERVER_KEY_BASE = 4400;

	// === DRIVING ===
	[Attribute("30", UIWidgets.Slider, "Waypoint completion radius [m]", "5 100 1", category: "Driving")]
	protected float m_fWaypointRadius;

	[Attribute("3000", UIWidgets.Slider, "Delay before the next waypoint is assigned after one is completed [ms]", "0 20000 100", category: "Driving")]
	protected int m_iWaypointDelay;

	[Attribute("1", UIWidgets.CheckBox, "Teleport a vehicle with a seated driver back to its group's spawn point when it is stuck", category: "Driving")]
	protected bool m_bResetStuckVehicles;

	[Attribute("30", UIWidgets.Slider, "A vehicle is stuck after this long within the stuck distance [s]", "5 300 1", category: "Driving")]
	protected float m_fStuckTimeS;

	[Attribute("5", UIWidgets.Slider, "Stuck distance [m]", "1 50 0.5", category: "Driving")]
	protected float m_fStuckDistanceM;

	[Attribute("200", UIWidgets.Slider, "Vehicle monitor interval (statistics, stuck detection) [ms]", "50 2000 50", category: "Driving")]
	protected int m_iMonitorIntervalMs;

	[Attribute("60000", UIWidgets.Slider, "Driving statistics log interval [ms], 0 = off", "0 600000 1000", category: "Driving")]
	protected int m_iStatsIntervalMs;

	// === ENVIRONMENT ===
	[Attribute("Overcast", UIWidgets.EditBox, "Initial weather state (StateName in WeatherStates.conf: Clear, Cloudy, Overcast, Rainy); empty = unchanged", category: "Environment")]
	protected string m_sInitialWeather;

	[Attribute("10", UIWidgets.Slider, "Initial hour of day, -1 = unchanged", "-1 23 1", category: "Environment")]
	protected int m_iInitialHour;

	// === START AND READINESS ===
	[Attribute("10000", UIWidgets.Slider, "Start this long after OnPostInit if OnGameModeStart never arrived [ms], 0 = no fallback", "0 60000 500", category: "Start")]
	protected int m_iFallbackStartMs;

	[Attribute("500", UIWidgets.Slider, "Driver readiness check interval [ms]", "100 5000 100", category: "Start")]
	protected int m_iReadinessPollMs;

	[Attribute("60000", UIWidgets.Slider, "Boarding window [ms], counted from 'Spawned N/M groups': the collector session starts as soon as every group has a seated driver, otherwise when the window ends with the groups seated by then", "5000 600000 1000", category: "Start")]
	protected int m_iBoardingWindowMs;

	[Attribute("1", UIWidgets.Slider, "Minimum number of groups with a seated driver for the session to start when the boarding window ends; set it to the number of AI groups for the old all-or-nothing gate", "1 10 1", category: "Start")]
	protected int m_iMinSeatedGroups;

	// === STATE ===
	protected bool m_bStarted;			//!< StartDirector ran (any machine), never cleared
	protected bool m_bActive;			//!< running on the authority and not stopped
	protected bool m_bReadinessDone;
	protected bool m_bReadinessArmed;	//!< the readiness poll was queued (after every group was attempted)
	protected float m_fBoardingWindowStartMs;	//!< world time when the boarding window opened ("Spawned N/M groups")
	protected bool m_bSessionStarted;	//!< this director started a collector session
	protected float m_fStartTimeMs;
	protected float m_fFallbackArmedMs;	//!< world time when FallbackStart was queued in OnPostInit
	protected int m_iNextGroupId;		//!< next group id the spawn tick attempts
	protected int m_iGroupsAttempted;
	protected int m_iGroupsFailed;
	protected int m_iTotalWaypointsCompleted;
	protected SCR_MLDataCollector m_Collector;
	protected ref array<ref SCR_MLDataCaptureWaypoint> m_aDriveTargets = {};	//!< marker records, m_Waypoint = null
	protected ref array<ref SCR_MLDataCaptureGroupState> m_aGroups = {};		//!< successfully started groups, ascending group id

	//------------------------------------------------------------------------------------------------
	override void OnPostInit(IEntity owner)
	{
		super.OnPostInit(owner);

		// OnPostInit also runs when the World Editor loads the layer: do nothing in edit mode.
		if (SCR_Global.IsEditMode(owner) || SCR_Global.IsEditMode())
			return;

		Print(string.Format("[DataCapture] Director initialized on '%1' (server: %2), waiting for OnGameModeStart", owner.GetName(), Replication.IsServer()), LogLevel.NORMAL);

		if (m_iFallbackStartMs > 0)
		{
			BaseWorld world = owner.GetWorld();
			if (world)
				m_fFallbackArmedMs = world.GetWorldTime();

			GetGame().GetCallqueue().CallLater(FallbackStart, m_iFallbackStartMs, false);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Called from SCR_BaseGameMode.OnGameStateChanged() when the state becomes GAME (on every machine).
	override void OnGameModeStart()
	{
		super.OnGameModeStart();

		if (m_bStarted)
		{
			Print("[DataCapture] OnGameModeStart received after the director had already started - ignored", LogLevel.NORMAL);
			return;
		}

		StartDirector("OnGameModeStart");
	}

	//------------------------------------------------------------------------------------------------
	protected void FallbackStart()
	{
		if (m_bStarted)
			return;

		// The call queue advances by frame timeslices, and the first frame after a long world load can carry a large
		// one. Fall back only once m_iFallbackStartMs of world time have passed since OnPostInit; until then re-queue.
		BaseWorld world = GetGame().GetWorld();
		if (world)
		{
			int remainingMs = m_iFallbackStartMs - (world.GetWorldTime() - m_fFallbackArmedMs);
			if (remainingMs > 0)
			{
				if (remainingMs < 100)
					remainingMs = 100;

				GetGame().GetCallqueue().CallLater(FallbackStart, remainingMs, false);
				return;
			}
		}

		Print(string.Format("[DataCapture] WARNING: OnGameModeStart not received within %1 ms", m_iFallbackStartMs), LogLevel.WARNING);
		StartDirector("FallbackStart");
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent: only the first call does anything.
	protected void StartDirector(string reason)
	{
		if (m_bStarted)
			return;

		if (!GetGame().InPlayMode())
			return;

		m_bStarted = true;
		GetGame().GetCallqueue().Remove(FallbackStart);

		if (!Replication.IsServer())
		{
			Print("[DataCapture] Director: not the server, nothing to do on this machine", LogLevel.NORMAL);
			return;
		}

		m_bActive = true;
		m_fStartTimeMs = GetGame().GetWorld().GetWorldTime();
		Print("[DataCapture] Director started (" + reason + ")", LogLevel.NORMAL);

		m_Collector = SCR_MLDataCollector.Cast(GetOwner().FindComponent(SCR_MLDataCollector));
		if (!m_Collector)
			Print("[DataCapture] WARNING: no SCR_MLDataCollector on the game mode entity - driving only, no capture session", LogLevel.WARNING);

		RegisterDebugMenus();
		InitializeEnvironment();
		CollectDriveTargets();
		Print(string.Format("[DataCapture] AI LOD: max LOD %1 (agents are deactivated there); driver LOD pin %2, vehicle observers %3", AIAgent.GetMaxLOD(), m_bPinDriverLOD, m_bVehicleObservers), LogLevel.NORMAL);

		if (m_aDriveTargets.IsEmpty())
		{
			Print(string.Format("[DataCapture] ERROR: no drive-target markers %1<1..%2> found - no groups spawned, no collector session", m_sDriveTargetPrefix, m_iMaxDriveTargets), LogLevel.ERROR);
			m_bReadinessDone = true;
			return;
		}

		// One repeating spawn tick: group k is attempted k * m_iSpawnStaggerMs after the start. The readiness poll is
		// queued by ReportSpawnProgress once every group has been attempted.
		ScriptCallQueue queue = GetGame().GetCallqueue();
		m_iNextGroupId = 1;
		queue.CallLater(SpawnNextGroup, m_iSpawnStaggerMs, true);
		queue.CallLater(MonitorVehicles, m_iMonitorIntervalMs, true);

		if (m_iStatsIntervalMs > 0)
			queue.CallLater(ReportDrivingStatistics, m_iStatsIntervalMs, true);

#ifdef ENABLE_DIAG
		queue.CallLater(UpdateDebugPanel, 16, true);
#endif
	}

	//------------------------------------------------------------------------------------------------
	//! Called on every machine when the game mode ends (SCR_BaseGameMode.OnGameStateChanged, POSTGAME).
	override void OnGameModeEnd(SCR_GameModeEndData data)
	{
		super.OnGameModeEnd(data);
		StopDirector(true);
	}

	//------------------------------------------------------------------------------------------------
	//! Called on every machine when the world ends (ArmaReforgerScripted.OnGameEnd -> SCR_BaseGameMode.OnGameEnd),
	//! before the entities are deleted, so the collector session is finalized while the world still exists.
	override void OnGameEnd()
	{
		super.OnGameEnd();
		StopDirector(true);
	}

	//------------------------------------------------------------------------------------------------
	override void OnDelete(IEntity owner)
	{
		// The collector finalizes itself in its own OnDelete; do not touch another component during deletion.
		StopDirector(false);
		super.OnDelete(owner);
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent. Removes every call the director queued (at most one per function, see the header) and
	//! unsubscribes from the groups. With stopCollector it also stops and finalizes the collector session.
	protected void StopDirector(bool stopCollector)
	{
		m_bActive = false;

		ArmaReforgerScripted game = GetGame();
		if (game)
		{
			ScriptCallQueue queue = game.GetCallqueue();
			if (queue)
			{
				queue.Remove(FallbackStart);
				queue.Remove(SpawnNextGroup);
				queue.Remove(MonitorVehicles);
				queue.Remove(CheckReadiness);
				queue.Remove(ReportDrivingStatistics);
#ifdef ENABLE_DIAG
				queue.Remove(UpdateDebugPanel);
#endif
			}
		}

		// During OnDelete (world teardown) only unsubscribe and drop the observers; the LOD pin dies with the agents.
		// No groups (edit mode, other machines, nothing spawned): nothing to release, so the world is not queried.
		ObserversSystem observers;
		if (!m_aGroups.IsEmpty())
			observers = GetObserversSystem();

		foreach (SCR_MLDataCaptureGroupState state : m_aGroups)
		{
			state.m_fNextAssignAtMs = -1;
			state.m_fNextSeatAttemptAtMs = -1;
			if (state.m_Group)
				state.m_Group.GetOnWaypointCompleted().Remove(OnWaypointCompleted);

			ReleaseGroup(state, observers, stopCollector);
		}

		if (stopCollector && m_Collector && m_Collector.IsCapturing())
		{
			Print("[DataCapture] Game mode or game ended - stopping the collector session", LogLevel.NORMAL);
			m_Collector.StopSession();
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void RegisterDebugMenus()
	{
#ifdef ENABLE_DIAG
		DiagMenu.RegisterMenu(SCR_DebugMenuID.DEBUGUI_MLDATACAPTURE_MENU, "ML Data Capture", "Game");
		DiagMenu.RegisterBool(SCR_DebugMenuID.DEBUGUI_MLDATACAPTURE_PANEL, "", "Show Driving Telemetry Panel", "ML Data Capture");
		DiagMenu.RegisterBool(SCR_DebugMenuID.DEBUGUI_MLDATACAPTURE_VEHICLES, "", "Show Vehicle Data", "ML Data Capture");
		DiagMenu.RegisterBool(SCR_DebugMenuID.DEBUGUI_MLDATACAPTURE_AI_INPUTS, "", "Show AI Inputs", "ML Data Capture");
#endif
	}

	//------------------------------------------------------------------------------------------------
	protected void InitializeEnvironment()
	{
		ChimeraWorld world = ChimeraWorld.CastFrom(GetGame().GetWorld());
		if (!world)
			return;

		TimeAndWeatherManagerEntity weather = world.GetTimeAndWeatherManager();
		if (!weather)
		{
			Print("[DataCapture] WARNING: no TimeAndWeatherManagerEntity in the world - time and weather unchanged", LogLevel.WARNING);
			return;
		}

		if (m_iInitialHour >= 0)
		{
			// GameLib/generated/Entities/BaseWeatherManagerEntity.c:280 (int, int, int, bool immediateChange)
			weather.SetHoursMinutesSeconds(m_iInitialHour, 0, 0, true);
			Print(string.Format("[DataCapture] Time set to %1:00", m_iInitialHour), LogLevel.NORMAL);
		}

		if (!m_sInitialWeather.IsEmpty())
		{
			// GameCode/World/TimeAndWeatherManagerEntity.c:265 (server only; looping, transition, state duration)
			weather.ForceWeatherTo(true, m_sInitialWeather, 0.1, 999999);
			Print("[DataCapture] Weather set to " + m_sInitialWeather, LogLevel.NORMAL);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Looks up <m_sDriveTargetPrefix>1..m_iMaxDriveTargets once. Missing markers are skipped; the others keep their id.
	protected void CollectDriveTargets()
	{
		m_aDriveTargets.Clear();

		BaseWorld world = GetGame().GetWorld();
		string missing;
		for (int targetId = 1; targetId <= m_iMaxDriveTargets; targetId++)
		{
			IEntity marker = world.FindEntityByName(m_sDriveTargetPrefix + targetId.ToString());
			if (!marker)
			{
				missing += " " + targetId.ToString();
				continue;
			}

			m_aDriveTargets.Insert(new SCR_MLDataCaptureWaypoint(targetId, marker.GetOrigin(), null));
		}

		Print(string.Format("[DataCapture] Found %1/%2 drive targets %3<id>", m_aDriveTargets.Count(), m_iMaxDriveTargets, m_sDriveTargetPrefix), LogLevel.NORMAL);
		if (!missing.IsEmpty())
			Print("[DataCapture] WARNING: missing drive-target ids:" + missing, LogLevel.WARNING);
	}

	//------------------------------------------------------------------------------------------------
	//! Repeating spawn tick: attempts the next group id and removes itself after the last one.
	protected void SpawnNextGroup()
	{
		if (!m_bActive || m_iNextGroupId > m_iMaxAIGroups)
		{
			GetGame().GetCallqueue().Remove(SpawnNextGroup);
			return;
		}

		int groupId = m_iNextGroupId;
		m_iNextGroupId++;
		if (m_iNextGroupId > m_iMaxAIGroups)
			GetGame().GetCallqueue().Remove(SpawnNextGroup);

		SpawnAIGroup(groupId);
	}

	//------------------------------------------------------------------------------------------------
	//! Starts group groupId: resolves and validates everything first, then spawns. Any failure skips the group.
	protected void SpawnAIGroup(int groupId)
	{
		if (!m_bActive)
			return;

		m_iGroupsAttempted++;
		BaseWorld world = GetGame().GetWorld();

		string spawnName = m_sSpawnPointPrefix + groupId.ToString();
		IEntity spawnPoint = world.FindEntityByName(spawnName);
		if (!spawnPoint)
		{
			FailGroup(groupId, "spawn point '" + spawnName + "' not found");
			return;
		}

		string vehicleName = m_sVehiclePrefix + groupId.ToString();
		IEntity vehicleEntity = world.FindEntityByName(vehicleName);
		if (!vehicleEntity)
		{
			FailGroup(groupId, "vehicle '" + vehicleName + "' not found (no fallback to another group's vehicle)");
			return;
		}

		Vehicle vehicle = Vehicle.Cast(vehicleEntity);
		if (!vehicle)
		{
			FailGroup(groupId, "'" + vehicleName + "' is not a Vehicle");
			return;
		}

		VehicleWheeledSimulation simulation = VehicleWheeledSimulation.Cast(vehicle.FindComponent(VehicleWheeledSimulation));
		if (!simulation)
		{
			FailGroup(groupId, "'" + vehicleName + "' has no VehicleWheeledSimulation");
			return;
		}

		// GameCode/Vehicle/Vehicle.c:104, the primary pilot (driver) compartment
		if (!vehicle.GetPilotCompartmentSlot())
		{
			FailGroup(groupId, "'" + vehicleName + "' has no pilot compartment");
			return;
		}

		Resource driverResource = Resource.Load(m_sDriverPrefab);
		if (!driverResource || !driverResource.IsValid())
		{
			FailGroup(groupId, "invalid driver prefab '" + m_sDriverPrefab + "'");
			return;
		}

		Resource waypointResource = Resource.Load(m_sWaypointPrefab);
		if (!waypointResource || !waypointResource.IsValid())
		{
			FailGroup(groupId, "invalid waypoint prefab '" + m_sWaypointPrefab + "'");
			return;
		}

		Resource groupResource = Resource.Load(m_sGroupPrefab);
		if (!groupResource || !groupResource.IsValid())
		{
			FailGroup(groupId, "invalid group prefab '" + m_sGroupPrefab + "'");
			return;
		}

		IEntity groupEntity = GetGame().SpawnEntityPrefab(groupResource, world, MakeSpawnParams(spawnPoint.GetOrigin()));
		SCR_AIGroup group = SCR_AIGroup.Cast(groupEntity);
		if (!group)
		{
			if (groupEntity)
				SCR_EntityHelper.DeleteEntityAndChildren(groupEntity);

			FailGroup(groupId, "group prefab '" + m_sGroupPrefab + "' did not spawn an SCR_AIGroup");
			return;
		}

		SCR_MLDataCaptureGroupState state = new SCR_MLDataCaptureGroupState(groupId, group, vehicle, simulation, spawnPoint);
		string waypointError = CreateWaypoints(state, waypointResource);
		if (!waypointError.IsEmpty())
		{
			DeleteGroupEntities(state);
			FailGroup(groupId, waypointError);
			return;
		}

		// Publish the record only now that all of its resources exist.
		m_aGroups.Insert(state);

		// The group should start empty: unit slots are queued as extra members on foot (SCR_AIGroup.EOnInit ->
		// RequestSpawn) that the group would also try to board.
		if (group.m_aUnitPrefabSlots && !group.m_aUnitPrefabSlots.IsEmpty())
			Print(string.Format("[DataCapture] WARNING: group %1: group prefab '%2' has %3 unit slots - they spawn on foot next to the seated driver; use an empty group prefab such as Group_US_Base", groupId, m_sGroupPrefab, group.m_aUnitPrefabSlots.Count()), LogLevel.WARNING);

		// Redundant once a driver is seated (seating registers the vehicle with the group), kept for walk-in boarding.
		array<string> vehicleNames = {vehicleName};
		group.AddVehiclesStatic(vehicleNames); // Game/Entities/SCR_AIGroup.c:1847, unchanged since 1.6
		group.GetOnWaypointCompleted().Insert(OnWaypointCompleted); // invoker params: AIWaypoint wp (server only)

		Print(string.Format("[DataCapture] Group %1 spawned at %2 for vehicle %3 with %4 waypoints", groupId, spawnName, vehicleName, state.m_aWaypoints.Count()), LogLevel.NORMAL);

		// Before the driver exists, so it is pinned as it joins (GetOnAgentAdded).
		PinGroupLOD(state);
		AddVehicleObserver(state);
		Print(string.Format("[DataCapture] Group %1: lod_pin=%2 (permanent LOD 0, max LOD prevented, importance CRITICAL), observer=%3 on %4 (key %5)", groupId, state.m_bLODPinned, state.m_bObserverAdded, vehicleName, OBSERVER_KEY_BASE + groupId), LogLevel.NORMAL);

		TrySeatDriver(state);
		ReportSpawnProgress();
	}

	//------------------------------------------------------------------------------------------------
	//! Keeps the group's driver at full AI LOD for the whole capture: the 1.9.0.10 vanilla logistic-driver pin
	//! (SCR_TransportUnitComponent.SetGroupLifecyclePinned), built only from calls that exist unchanged in 1.8.0.13.
	//! CRITICAL importance keeps ChimeraAIWorld eviction from picking the group, and a permanent LOD pin also makes
	//! SCR_AIGroup.HasPermanentLodMember hold it. Runs before the driver exists; GetOnAgentAdded pins it as it joins.
	protected void PinGroupLOD(SCR_MLDataCaptureGroupState state)
	{
		SCR_AIGroup group = state.m_Group;
		if (!m_bPinDriverLOD || !group || state.m_bLODPinned)
			return;

		state.m_bLODPinned = true;
		state.m_iImportanceBeforePin = group.GetImportance();
		group.SetImportance(SCR_EAISpawnImportance.CRITICAL);

		// The group is an AIAgent too and its behaviour tree runs the waypoints. PreventMaxLOD does not lift an agent
		// that already is at max LOD, so step it down first (SCR_ResupplyTaskSolver.PreventMaxLOD, 1.8).
		int maxLOD = AIAgent.GetMaxLOD();
		if (group.GetLOD() == maxLOD)
			group.SetLOD(maxLOD - 1);

		group.PreventMaxLOD();
		group.GetOnAgentAdded().Insert(OnDriverAgentAdded);
		PinAgents(group);
	}

	//------------------------------------------------------------------------------------------------
	protected void PinAgents(SCR_AIGroup group)
	{
		if (!group)
			return;

		array<AIAgent> agents = {};
		group.GetAgents(agents);
		foreach (AIAgent agent : agents)
		{
			if (agent)
				agent.SetPermanentLOD(0);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! SCR_AIGroup.GetOnAgentAdded() handler of pinned groups; the invoker passes only the agent (SCR_AIGroup.OnAgentAdded).
	protected void OnDriverAgentAdded(AIAgent agent)
	{
		if (agent)
			agent.SetPermanentLOD(0);
	}

	//------------------------------------------------------------------------------------------------
	//! Puts a temporary local observer on the group's vehicle. With an entity, x/z are a local offset, so 0, 0 is the
	//! car itself (Game/generated/System/ObserversSystem.c). The ManualCamera is the only other observer, and once a
	//! session starts SCR_MLScreenshotCapture attaches it to the first registered car.
	protected void AddVehicleObserver(SCR_MLDataCaptureGroupState state)
	{
		if (!m_bVehicleObservers || state.m_bObserverAdded || !state.m_Vehicle)
			return;

		ObserversSystem observers = GetObserversSystem();
		if (!observers)
		{
			Print(string.Format("[DataCapture] WARNING: group %1: no ObserversSystem in this world - no observer on %2, a car far from the camera may not be simulated", state.m_iGroupId, state.m_Vehicle.GetName()), LogLevel.WARNING);
			return;
		}

		observers.InsertObserverSP(OBSERVER_KEY_BASE + state.m_iGroupId, 0, 0, state.m_Vehicle);
		state.m_bObserverAdded = true;
	}

	//------------------------------------------------------------------------------------------------
	protected ObserversSystem GetObserversSystem()
	{
		ArmaReforgerScripted game = GetGame();
		if (!game)
			return null;

		ChimeraWorld world = ChimeraWorld.CastFrom(game.GetWorld());
		if (!world)
			return null;

		return ObserversSystem.Cast(world.FindSystem(ObserversSystem));
	}

	//------------------------------------------------------------------------------------------------
	//! Undoes PinGroupLOD (only with restoreLOD; not during world teardown) and removes the vehicle observer. Idempotent.
	protected void ReleaseGroup(SCR_MLDataCaptureGroupState state, ObserversSystem observers, bool restoreLOD)
	{
		SCR_AIGroup group = state.m_Group;
		if (state.m_bLODPinned)
		{
			state.m_bLODPinned = false;
			if (group)
			{
				group.GetOnAgentAdded().Remove(OnDriverAgentAdded);
				if (restoreLOD)
				{
					array<AIAgent> agents = {};
					group.GetAgents(agents);
					foreach (AIAgent agent : agents)
					{
						if (agent)
							agent.SetPermanentLOD(-1);
					}

					group.AllowMaxLOD();
					if (state.m_iImportanceBeforePin != -1)
						group.SetImportance(state.m_iImportanceBeforePin);
				}
			}
		}

		if (state.m_bObserverAdded)
		{
			state.m_bObserverAdded = false;
			if (observers)
				observers.RemoveObserverSP(OBSERVER_KEY_BASE + state.m_iGroupId);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! One-line diagnostics of a group: how its driver boarded, agent count, dormancy, whether an observer is within
	//! 1 km, the group's LOD, per-agent LOD / max / permanent LOD / activation, and whether the pilot seat is taken.
	protected string DescribeGroup(SCR_MLDataCaptureGroupState state)
	{
		string text = "boarding=" + typename.EnumToString(SCR_EMLDataCaptureBoarding, state.m_eBoarding);
		SCR_AIGroup group = state.m_Group;
		if (group)
		{
			text += string.Format(" agents=%1 dormant=%2 observer_1km=%3 group_lod=%4", group.GetAgentsCount(), group.IsDormant(), group.HasObserverInRange(1000), group.GetLOD());

			array<AIAgent> agents = {};
			group.GetAgents(agents);
			foreach (AIAgent agent : agents)
			{
				if (agent)
					text += string.Format(" [lod=%1/%2 perm=%3 active=%4]", agent.GetLOD(), AIAgent.GetMaxLOD(), agent.GetPermanentLOD(), agent.IsAIActivated());
			}
		}
		else
		{
			text += " group=deleted";
		}

		bool pilot;
		if (state.m_Vehicle)
			pilot = state.m_Vehicle.GetPilot() != null;

		return text + string.Format(" pilot=%1 lod_pin=%2 observer=%3", pilot, state.m_bLODPinned, state.m_bObserverAdded);
	}

	//------------------------------------------------------------------------------------------------
	//! One direct seating attempt: spawns m_sDriverPrefab straight into the vehicle's pilot seat and adds it to the
	//! group. BaseCompartmentSlot.SpawnCharacterInCompartment (GameCode/Vehicle/BaseCompartmentSlot.c:328, the Game
	//! Master "crewed" and ScenarioFramework QRF path) spawns the character, activates its AI, seats it with
	//! CompartmentAccessComponent.GetInVehicle(forceTeleport = true) and calls AIGroup.AddAgent; it returns null and
	//! deletes the character when the seat cannot be taken. Walking is not needed, so the distance to the camera does
	//! not matter. On failure the next attempt is due m_iSeatRetryMs later (run by MonitorVehicles); after
	//! m_iSeatAttempts failures the group falls back to walk-in boarding.
	protected void TrySeatDriver(SCR_MLDataCaptureGroupState state)
	{
		state.m_fNextSeatAttemptAtMs = -1;
		if (!m_bActive || state.m_eBoarding != SCR_EMLDataCaptureBoarding.PENDING)
			return;

		if (!state.m_Group || !state.m_Vehicle)
		{
			state.m_eBoarding = SCR_EMLDataCaptureBoarding.FAILED;
			Print(string.Format("[DataCapture] ERROR: group %1: its group or vehicle was deleted before a driver was seated - no driver", state.m_iGroupId), LogLevel.ERROR);
			return;
		}

		state.m_iSeatAttemptsMade++;
		string vehicleName = state.m_Vehicle.GetName();
		PilotCompartmentSlot pilotSlot = state.m_Vehicle.GetPilotCompartmentSlot();
		IEntity driver;
		if (pilotSlot)
		{
			// inout: with a non-null group the character joins it and no new group is created
			AIGroup aiGroup = state.m_Group;
			driver = pilotSlot.SpawnCharacterInCompartment(m_sDriverPrefab, aiGroup);
		}

		if (driver)
		{
			state.m_Driver = driver;
			state.m_eBoarding = SCR_EMLDataCaptureBoarding.DIRECT;
			if (state.m_bLODPinned)
				PinAgents(state.m_Group);	// again after the join, in case the agent-added event came first

			Print(string.Format("[DataCapture] Group %1: driver spawned in the pilot seat of %2 (direct seating, attempt %3/%4) %5", state.m_iGroupId, vehicleName, state.m_iSeatAttemptsMade, m_iSeatAttempts, DescribeGroup(state)), LogLevel.NORMAL);
			StartDriving(state);
			return;
		}

		string reason = "no pilot compartment";
		if (pilotSlot)
			reason = string.Format("pilot seat occupant=%1 accessible=%2", pilotSlot.GetOccupant(), pilotSlot.IsCompartmentAccessible());

		if (state.m_iSeatAttemptsMade < m_iSeatAttempts)
		{
			state.m_fNextSeatAttemptAtMs = GetGame().GetWorld().GetWorldTime() + m_iSeatRetryMs;
			Print(string.Format("[DataCapture] WARNING: group %1: direct seating attempt %2/%3 into %4 failed (%5) - retrying in %6 ms", state.m_iGroupId, state.m_iSeatAttemptsMade, m_iSeatAttempts, vehicleName, reason, m_iSeatRetryMs), LogLevel.WARNING);
			return;
		}

		Print(string.Format("[DataCapture] WARNING: group %1: direct seating into %2 failed %3 times (%4) - falling back to walk-in boarding", state.m_iGroupId, vehicleName, state.m_iSeatAttemptsMade, reason), LogLevel.WARNING);
		StartWalkInBoarding(state);
	}

	//------------------------------------------------------------------------------------------------
	//! Fallback after direct seating failed: spawns the driver on foot at the group's spawn point and adds it to the
	//! group (SCR_AIGroup.AddAIEntityToGroup activates its AI), then starts the waypoints. The group owns its vehicle
	//! through AddVehiclesStatic, so the vanilla group AI walks the driver to the car and gets in before it drives.
	protected void StartWalkInBoarding(SCR_MLDataCaptureGroupState state)
	{
		IEntity driver;
		Resource driverResource = Resource.Load(m_sDriverPrefab);
		if (driverResource && driverResource.IsValid() && state.m_SpawnPoint)
			driver = GetGame().SpawnEntityPrefab(driverResource, GetGame().GetWorld(), MakeSpawnParams(state.m_SpawnPoint.GetOrigin()));

		if (!driver || !state.m_Group || !state.m_Group.AddAIEntityToGroup(driver))
		{
			if (driver)
				SCR_EntityHelper.DeleteEntityAndChildren(driver);

			state.m_eBoarding = SCR_EMLDataCaptureBoarding.FAILED;
			Print(string.Format("[DataCapture] ERROR: group %1: walk-in fallback failed: could not spawn '%2' at its spawn point or add it to the group - no driver", state.m_iGroupId, m_sDriverPrefab), LogLevel.ERROR);
			return;
		}

		state.m_Driver = driver;
		state.m_eBoarding = SCR_EMLDataCaptureBoarding.WALK_IN;
		if (state.m_bLODPinned)
			PinAgents(state.m_Group);

		Print(string.Format("[DataCapture] WARNING: group %1: driver spawned on foot at %2 - it has to walk to %3 and get in (walk-in fallback) %4", state.m_iGroupId, state.m_SpawnPoint.GetName(), state.m_Vehicle.GetName(), DescribeGroup(state)), LogLevel.WARNING);
		StartDriving(state);
	}

	//------------------------------------------------------------------------------------------------
	//! Assigns the group's first waypoint once it has a driver.
	protected void StartDriving(SCR_MLDataCaptureGroupState state)
	{
		if (state.m_bDrivingStarted)
			return;

		state.m_bDrivingStarted = true;
		AssignNextWaypoint(state);
	}

	//------------------------------------------------------------------------------------------------
	//! Spawns one waypoint per drive target for this group.
	//! \return empty string on success, otherwise the reason (the waypoints spawned so far are left in state for cleanup)
	protected string CreateWaypoints(SCR_MLDataCaptureGroupState state, Resource waypointResource)
	{
		BaseWorld world = GetGame().GetWorld();
		foreach (SCR_MLDataCaptureWaypoint target : m_aDriveTargets)
		{
			IEntity entity = GetGame().SpawnEntityPrefab(waypointResource, world, MakeSpawnParams(target.m_vPosition));
			AIWaypoint waypoint = AIWaypoint.Cast(entity);
			if (!waypoint)
			{
				if (entity)
					SCR_EntityHelper.DeleteEntityAndChildren(entity);

				return "waypoint prefab '" + m_sWaypointPrefab + "' did not spawn an AIWaypoint";
			}

			waypoint.SetCompletionRadius(m_fWaypointRadius);
			waypoint.SetName(string.Format("DC_G%1_WP%2", state.m_iGroupId, target.m_iTargetId)); // unique: SetName fails on duplicates in 1.8
			state.m_aWaypoints.Insert(new SCR_MLDataCaptureWaypoint(target.m_iTargetId, target.m_vPosition, waypoint));
		}

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	protected void DeleteGroupEntities(SCR_MLDataCaptureGroupState state)
	{
		foreach (SCR_MLDataCaptureWaypoint record : state.m_aWaypoints)
		{
			if (record.m_Waypoint)
				SCR_EntityHelper.DeleteEntityAndChildren(record.m_Waypoint);
		}
		state.m_aWaypoints.Clear();

		if (state.m_Group)
			SCR_EntityHelper.DeleteEntityAndChildren(state.m_Group);
	}

	//------------------------------------------------------------------------------------------------
	protected void FailGroup(int groupId, string reason)
	{
		m_iGroupsFailed++;
		Print(string.Format("[DataCapture] ERROR: group %1 skipped: %2", groupId, reason), LogLevel.ERROR);
		ReportSpawnProgress();
	}

	//------------------------------------------------------------------------------------------------
	//! After the last group was attempted: logs "Spawned N/M groups" once, opens the boarding window and queues the
	//! readiness poll, so the log reads director start -> spawned N/M groups -> driver seated lines -> session start.
	protected void ReportSpawnProgress()
	{
		if (m_iGroupsAttempted < m_iMaxAIGroups || m_bReadinessArmed)
			return;

		m_bReadinessArmed = true;
		m_fBoardingWindowStartMs = GetGame().GetWorld().GetWorldTime();
		float windowS = m_iBoardingWindowMs / 1000.0;
		Print(string.Format("[DataCapture] Spawned %1/%2 groups - boarding window %3 s: the session starts once all %2 drivers are seated, or when the window ends with at least %4 seated", m_aGroups.Count(), m_iMaxAIGroups, windowS.ToString(-1, 0), GetMinSeatedGroups()), LogLevel.NORMAL);

		if (m_bActive && !m_bReadinessDone)
			GetGame().GetCallqueue().CallLater(CheckReadiness, m_iReadinessPollMs, true);
	}

	//------------------------------------------------------------------------------------------------
	//! m_iMinSeatedGroups limited to 1..m_iMaxAIGroups: a session needs at least one vehicle.
	protected int GetMinSeatedGroups()
	{
		return Math.ClampInt(m_iMinSeatedGroups, 1, m_iMaxAIGroups);
	}

	//------------------------------------------------------------------------------------------------
	protected SCR_MLDataCaptureGroupState FindGroup(int groupId)
	{
		foreach (SCR_MLDataCaptureGroupState state : m_aGroups)
		{
			if (state.m_iGroupId == groupId)
				return state;
		}

		return null;
	}

	//------------------------------------------------------------------------------------------------
	//! Gives the group a new waypoint: never the current one and never one within 2x the completion radius of the
	//! vehicle or of the current target. If no target qualifies, the group gets none and MonitorVehicles retries
	//! (an excluded target would be completed without the vehicle moving).
	protected void AssignNextWaypoint(SCR_MLDataCaptureGroupState state)
	{
		if (!m_bActive || !state || !state.m_Group)
			return;

		state.m_fNextAssignAtMs = -1;
		int groupId = state.m_iGroupId;
		int count = state.m_aWaypoints.Count();
		int current = state.m_iCurrentWaypoint;
		float exclusion = 2 * m_fWaypointRadius;

		vector here = state.m_vLastPosition;
		if (state.m_Vehicle)
			here = state.m_Vehicle.GetOrigin();

		array<int> candidates = {};
		for (int i = 0; i < count; i++)
		{
			if (i == current)
				continue;

			vector position = state.m_aWaypoints[i].m_vPosition;
			if (vector.DistanceXZ(position, here) < exclusion)
				continue;

			if (current >= 0 && vector.DistanceXZ(position, state.m_aWaypoints[current].m_vPosition) < exclusion)
				continue;

			candidates.Insert(i);
		}

		if (candidates.IsEmpty())
		{
			int retryMs = m_iWaypointDelay;
			if (retryMs < 1000)
				retryMs = 1000;

			state.m_fNextAssignAtMs = GetGame().GetWorld().GetWorldTime() + retryMs;
			if (!state.m_bNoTargetWarned)
			{
				state.m_bNoTargetWarned = true;
				Print(string.Format("[DataCapture] WARNING: group %1: no drive target is %2 m or more away from the vehicle and the current target - no new waypoint, retrying every %3 ms", groupId, exclusion, retryMs), LogLevel.WARNING);
			}

			return;
		}

		state.m_bNoTargetWarned = false;
		int next = candidates[Math.RandomIntInclusive(0, candidates.Count() - 1)];
		int previousTargetId = state.GetCurrentTargetId();
		SCR_MLDataCaptureWaypoint record = state.m_aWaypoints[next];

		state.m_Group.AddWaypoint(record.m_Waypoint);
		state.m_iCurrentWaypoint = next;

		if (state.m_bRegistered && m_Collector)
			m_Collector.SetVehicleWaypoint(state.m_Vehicle, record.m_iTargetId, record.m_vPosition);

		string previous = "none";
		if (previousTargetId > 0)
			previous = previousTargetId.ToString();

		float distance = vector.Distance(here, record.m_vPosition);
		Print(string.Format("[DataCapture] Group %1 waypoint %2 -> %3 (%4%3, %5 m away)", groupId, previous, record.m_iTargetId, m_sDriveTargetPrefix, distance.ToString(-1, 0)), LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	protected void OnWaypointCompleted(AIWaypoint completedWaypoint)
	{
		if (!m_bActive)
			return;

		foreach (SCR_MLDataCaptureGroupState state : m_aGroups)
		{
			int index = state.FindWaypoint(completedWaypoint);
			if (index < 0)
				continue;

			m_iTotalWaypointsCompleted++;
			state.m_iWaypointsCompleted++;

			if (state.m_Group)
				state.m_Group.RemoveWaypoint(completedWaypoint);

			// One pending assignment per group, run by MonitorVehicles; a repeated completion event only re-arms it.
			state.m_fNextAssignAtMs = GetGame().GetWorld().GetWorldTime() + m_iWaypointDelay;
			return;
		}
	}

	//------------------------------------------------------------------------------------------------
	//! \return true when a living character sits in the vehicle's pilot seat
	protected bool HasLivingDriver(Vehicle vehicle)
	{
		if (!vehicle)
			return false;

		// GameCode/Vehicle/Vehicle.c:114; alive check as in Game/AI/ScriptedNodes/Vehicles/SCR_AIDecoTestIsDriverAliveInsideVehicle.c
		ChimeraCharacter driver = ChimeraCharacter.Cast(vehicle.GetPilot());
		if (!driver)
			return false;

		CharacterControllerComponent controller = driver.GetCharacterController();
		return controller && controller.GetLifeState() == ECharacterLifeState.ALIVE;
	}

	//------------------------------------------------------------------------------------------------
	//! Re-reads whether a living driver sits in the group's pilot seat and remembers when one was first seen.
	protected void UpdateOccupancy(SCR_MLDataCaptureGroupState state, float elapsedS)
	{
		state.m_bDriverSeated = HasLivingDriver(state.m_Vehicle);
		if (state.m_bDriverSeated && state.m_fFirstSeatedS < 0)
			state.m_fFirstSeatedS = elapsedS;
	}

	//------------------------------------------------------------------------------------------------
	//! Repeating until done, queued once every group was attempted ("Spawned N/M groups" opens the boarding window).
	//! Logs each seated driver once and starts the collector session (StartCaptureSession) as soon as every configured
	//! group has a seated driver, or when the window ends with at least GetMinSeatedGroups() seated. Otherwise, or as
	//! soon as the minimum can no longer be reached, it logs an ERROR with the missing groups and starts no session.
	protected void CheckReadiness()
	{
		if (!m_bActive || m_bReadinessDone)
		{
			GetGame().GetCallqueue().Remove(CheckReadiness);
			return;
		}

		float nowMs = GetGame().GetWorld().GetWorldTime();
		float elapsedS = (nowMs - m_fStartTimeMs) / 1000.0;
		int seated;
		int canSeat;	// seated, or a driver still pending / walking in (boarding not FAILED)
		foreach (SCR_MLDataCaptureGroupState state : m_aGroups)
		{
			UpdateOccupancy(state, elapsedS);
			if (!state.m_bDriverSeated)
			{
				if (state.m_eBoarding != SCR_EMLDataCaptureBoarding.FAILED)
					canSeat++;

				continue;
			}

			seated++;
			canSeat++;
			if (!state.m_bSeatedLogged)
			{
				state.m_bSeatedLogged = true;
				Print(string.Format("[DataCapture] Group %1: driver seated in %2 (first seen %3 s after the director start) %4", state.m_iGroupId, state.m_Vehicle.GetName(), state.m_fFirstSeatedS.ToString(-1, 1), DescribeGroup(state)), LogLevel.NORMAL);
			}
		}

		if (seated >= m_iMaxAIGroups)
		{
			StartCaptureSession(string.Format("All %1 drivers seated after %2 s", seated, elapsedS.ToString(-1, 1)));
			return;
		}

		// Groups that failed to start or to get a driver never seat: stop waiting once the minimum is out of reach.
		int minSeated = GetMinSeatedGroups();
		if (canSeat < minSeated)
		{
			FinishReadiness(string.Format("[DataCapture] ERROR: only %1 of %2 groups can still get a seated driver (%3 failed to start), fewer than the minimum of %4 - collector session NOT started", canSeat, m_iMaxAIGroups, m_iGroupsFailed, minSeated));
			LogMissingGroups(LogLevel.ERROR);
			return;
		}

		if (nowMs - m_fBoardingWindowStartMs < m_iBoardingWindowMs)
			return;

		float windowS = m_iBoardingWindowMs / 1000.0;
		string missing = MissingGroupIds();
		if (seated < minSeated)
		{
			string outcome = string.Format("%1/%2 drivers seated, fewer than the minimum of %3", seated, m_iMaxAIGroups, minSeated);
			if (seated == 0)
				outcome = string.Format("no seated driver (0/%1)", m_iMaxAIGroups);

			FinishReadiness(string.Format("[DataCapture] ERROR: boarding window of %1 s ended with %2 (groups without a seated driver:%3) - collector session NOT started", windowS.ToString(-1, 0), outcome, missing));
			LogMissingGroups(LogLevel.ERROR);
			return;
		}

		Print(string.Format("[DataCapture] WARNING: boarding window of %1 s ended with %2/%3 drivers seated (minimum %4) - starting the session without groups:%5", windowS.ToString(-1, 0), seated, m_iMaxAIGroups, minSeated, missing), LogLevel.WARNING);
		LogMissingGroups(LogLevel.WARNING);
		StartCaptureSession(string.Format("%1/%2 drivers seated after %3 s", seated, m_iMaxAIGroups, elapsedS.ToString(-1, 1)));
	}

	//------------------------------------------------------------------------------------------------
	//! Ends the boarding window: registers every vehicle with a seated driver with the collector, in group-id order
	//! (vehicle_id = registration index, k - 1 when every group is seated), then starts the session. Groups without a
	//! seated driver now stay out of this session; MonitorVehicles logs it if one seats later.
	protected void StartCaptureSession(string summary)
	{
		m_bReadinessDone = true;
		GetGame().GetCallqueue().Remove(CheckReadiness);

		if (!m_Collector)
		{
			Print("[DataCapture] " + summary + " (no collector, no session)", LogLevel.NORMAL);
			return;
		}

		array<SCR_MLDataCaptureGroupState> seatedGroups = {};
		string vehicles;
		foreach (SCR_MLDataCaptureGroupState state : m_aGroups)
		{
			if (!state.m_bDriverSeated || state.m_bRegistered)
				continue;

			vehicles += string.Format(" V%1=%2", seatedGroups.Count(), state.m_Vehicle.GetName());
			seatedGroups.Insert(state);
		}

		Print(string.Format("[DataCapture] %1 - starting the collector session with %2 vehicles:%3", summary, seatedGroups.Count(), vehicles), LogLevel.NORMAL);
		foreach (SCR_MLDataCaptureGroupState seatedState : seatedGroups)
		{
			m_Collector.RegisterVehicle(seatedState.m_Vehicle);
			seatedState.m_bRegistered = true;

			int targetId = seatedState.GetCurrentTargetId();
			if (targetId > 0)
				m_Collector.SetVehicleWaypoint(seatedState.m_Vehicle, targetId, seatedState.m_aWaypoints[seatedState.m_iCurrentWaypoint].m_vPosition);
		}

		if (!m_Collector.StartSession(string.Empty, m_sInitialWeather, m_iInitialHour))
		{
			Print("[DataCapture] ERROR: SCR_MLDataCollector.StartSession() failed - no session", LogLevel.ERROR);
			return;
		}

		m_bSessionStarted = true;
	}

	//------------------------------------------------------------------------------------------------
	//! \return " 3 4 5": the configured group ids without a seated driver (including groups that failed to start)
	protected string MissingGroupIds()
	{
		string missing;
		for (int groupId = 1; groupId <= m_iMaxAIGroups; groupId++)
		{
			SCR_MLDataCaptureGroupState state = FindGroup(groupId);
			if (!state || !state.m_bDriverSeated)
				missing += " " + groupId.ToString();
		}

		return missing;
	}

	//------------------------------------------------------------------------------------------------
	//! One line per configured group without a seated driver: why it failed to start, or its DescribeGroup state.
	protected void LogMissingGroups(LogLevel level)
	{
		for (int groupId = 1; groupId <= m_iMaxAIGroups; groupId++)
		{
			SCR_MLDataCaptureGroupState state = FindGroup(groupId);
			if (!state)
			{
				Print(string.Format("[DataCapture]   group %1: failed to start (see 'group %1 skipped')", groupId), level);
				continue;
			}

			if (state.m_bDriverSeated)
				continue;

			string vehicleName = "deleted vehicle";
			if (state.m_Vehicle)
				vehicleName = state.m_Vehicle.GetName();

			Print(string.Format("[DataCapture]   group %1 (%2): no seated driver, %3", groupId, vehicleName, DescribeGroup(state)), level);
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void FinishReadiness(string errorMessage)
	{
		m_bReadinessDone = true;
		GetGame().GetCallqueue().Remove(CheckReadiness);
		Print(errorMessage, LogLevel.ERROR);
	}

	//------------------------------------------------------------------------------------------------
	//! Repeating: distance and speed statistics, and stuck detection for vehicles with a seated driver.
	protected void MonitorVehicles()
	{
		if (!m_bActive)
			return;

		float nowMs = GetGame().GetWorld().GetWorldTime();
		float elapsedS = (nowMs - m_fStartTimeMs) / 1000.0;
		foreach (SCR_MLDataCaptureGroupState state : m_aGroups)
		{
			// Occupancy for stuck recovery is re-read here, so it does not depend on the readiness poll having run
			// or still running, and it drops when the driver leaves or dies.
			UpdateOccupancy(state, elapsedS);

			// A driver seated after the boarding window is not added to a running session: the collector's vehicle
			// ids and session_info.txt are fixed at StartSession. Log it once; the car keeps driving.
			if (m_bReadinessDone && state.m_bDriverSeated && !state.m_bSeatedLogged)
			{
				state.m_bSeatedLogged = true;
				string session = "no collector session is running";
				if (m_bSessionStarted)
					session = "not added to the running collector session (its vehicle ids and session_info.txt are fixed at its start)";

				Print(string.Format("[DataCapture] WARNING: group %1: driver seated in %2 %3 s after the director start, after the boarding window - %4; the car keeps driving", state.m_iGroupId, state.m_Vehicle.GetName(), elapsedS.ToString(-1, 1), session), LogLevel.WARNING);
			}

			if (state.m_fNextSeatAttemptAtMs >= 0 && nowMs >= state.m_fNextSeatAttemptAtMs)
				TrySeatDriver(state);

			if (state.m_fNextAssignAtMs >= 0 && nowMs >= state.m_fNextAssignAtMs)
				AssignNextWaypoint(state);

			if (!state.m_Vehicle || !state.m_Simulation)
				continue;

			float dtS = 0;
			if (state.m_fLastMonitorTimeMs >= 0)
				dtS = (nowMs - state.m_fLastMonitorTimeMs) / 1000.0;
			state.m_fLastMonitorTimeMs = nowMs;

			vector position = state.m_Vehicle.GetOrigin();
			state.m_fDistance += vector.Distance(position, state.m_vLastPosition);
			state.m_vLastPosition = position;

			float speed = state.m_Simulation.GetSpeedKmh(); // VehicleBaseSimulation.GetSpeedKmh in 1.8
			if (dtS > 0)
				state.m_fAcceleration = (speed - state.m_fLastSpeed) / dtS;
			state.m_fLastSpeed = speed;

			if (speed > state.m_fMaxSpeed)
				state.m_fMaxSpeed = speed;

			float rpm = state.m_Simulation.EngineGetRPM();
			if (rpm > state.m_fMaxRPM)
				state.m_fMaxRPM = rpm;

			CheckStuck(state, position, dtS);
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void CheckStuck(SCR_MLDataCaptureGroupState state, vector position, float dtS)
	{
		// Only while a living driver sits in the pilot seat: a car waiting for its crew is parked, not stuck.
		if (!m_bResetStuckVehicles || !state.m_bDriverSeated)
		{
			state.m_fStuckTimeS = 0;
			state.m_vStuckCheckPosition = position;
			return;
		}

		if (vector.Distance(position, state.m_vStuckCheckPosition) >= m_fStuckDistanceM)
		{
			state.m_fStuckTimeS = 0;
			state.m_vStuckCheckPosition = position;
			return;
		}

		state.m_fStuckTimeS += dtS;
		if (state.m_fStuckTimeS >= m_fStuckTimeS)
			ResetStuckVehicle(state);
	}

	//------------------------------------------------------------------------------------------------
	//! Teleports the vehicle 2 m above its own group's spawn point.
	protected void ResetStuckVehicle(SCR_MLDataCaptureGroupState state)
	{
		state.m_fStuckTimeS = 0;
		if (!state.m_Vehicle || !state.m_SpawnPoint)
			return;

		state.m_iStuckResets++;
		Print(string.Format("[DataCapture] Group %1: vehicle %2 stuck for %3 s - resetting to %4 (reset %5)", state.m_iGroupId, state.m_Vehicle.GetName(), m_fStuckTimeS, state.m_SpawnPoint.GetName(), state.m_iStuckResets), LogLevel.WARNING);

		vector transform[4];
		state.m_SpawnPoint.GetWorldTransform(transform);
		vector position = transform[3];
		position[1] = position[1] + 2.0;
		transform[3] = position;
		state.m_Vehicle.SetWorldTransform(transform);

		state.m_vStuckCheckPosition = position;
		state.m_vLastPosition = position;
	}

	//------------------------------------------------------------------------------------------------
	protected void ReportDrivingStatistics()
	{
		if (!m_bActive)
			return;

		float minutes = (GetGame().GetWorld().GetWorldTime() - m_fStartTimeMs) / 60000.0;

		Print("=== ML DATA CAPTURE DRIVING STATISTICS ===", LogLevel.NORMAL);
		Print(string.Format("Director time: %1 min, groups: %2/%3, waypoints completed: %4", minutes.ToString(-1, 1), m_aGroups.Count(), m_iMaxAIGroups, m_iTotalWaypointsCompleted), LogLevel.NORMAL);
		foreach (SCR_MLDataCaptureGroupState state : m_aGroups)
		{
			Print(string.Format("Group %1: distance=%2 m max_speed=%3 km/h waypoints=%4 stuck_resets=%5 target=%6 driver_seated=%7 %8",
				state.m_iGroupId, state.m_fDistance.ToString(-1, 1), state.m_fMaxSpeed.ToString(-1, 1), state.m_iWaypointsCompleted,
				state.m_iStuckResets, state.GetCurrentTargetId(), state.m_bDriverSeated, DescribeGroup(state)), LogLevel.NORMAL);
		}
		Print("==========================================", LogLevel.NORMAL);
	}

#ifdef ENABLE_DIAG
	//------------------------------------------------------------------------------------------------
	protected void UpdateDebugPanel()
	{
		if (!DiagMenu.GetBool(SCR_DebugMenuID.DEBUGUI_MLDATACAPTURE_PANEL))
			return;

		float simTimeS = (GetGame().GetWorld().GetWorldTime() - m_fStartTimeMs) / 1000.0;

		DbgUI.Begin("ML Data Capture - Director");
		DbgUI.Text(string.Format("Time: %1 s  Groups: %2/%3  Waypoints completed: %4", simTimeS.ToString(-1, 1), m_aGroups.Count(), m_iMaxAIGroups, m_iTotalWaypointsCompleted));

		if (DiagMenu.GetBool(SCR_DebugMenuID.DEBUGUI_MLDATACAPTURE_VEHICLES))
		{
			bool showInputs = DiagMenu.GetBool(SCR_DebugMenuID.DEBUGUI_MLDATACAPTURE_AI_INPUTS);
			foreach (SCR_MLDataCaptureGroupState state : m_aGroups)
			{
				VehicleWheeledSimulation sim = state.m_Simulation;
				if (!sim || !state.m_Vehicle)
					continue;

				float speed = sim.GetSpeedKmh();
				float rpm = sim.EngineGetRPM();
				string engine = "OFF";
				if (sim.EngineIsOn())
					engine = "ON";

				DbgUI.Text(string.Format("--- GROUP %1 (%2) target %3 seated %4 ---", state.m_iGroupId, state.m_Vehicle.GetName(), state.GetCurrentTargetId(), state.m_bDriverSeated));
				DbgUI.Text(string.Format("Speed: %1 km/h  Accel: %2 km/h/s  Engine: %3 (%4 RPM)  Gear: %5", speed.ToString(-1, 1), state.m_fAcceleration.ToString(-1, 1), engine, rpm.ToString(-1, 0), sim.GetGear()));
				if (state.m_fStuckTimeS > 5.0)
					DbgUI.Text(string.Format("STUCK WARNING: %1 s", state.m_fStuckTimeS.ToString(-1, 1)));

				if (showInputs)
					DbgUI.Text(string.Format("Steering: %1  Throttle: %2  Brake: %3", sim.GetSteering().ToString(-1, 2), sim.GetThrottle().ToString(-1, 2), sim.GetBrake().ToString(-1, 2)));
			}
		}

		DbgUI.End();
	}
#endif

	//------------------------------------------------------------------------------------------------
	protected EntitySpawnParams MakeSpawnParams(vector position)
	{
		EntitySpawnParams params = new EntitySpawnParams();
		params.TransformMode = ETransformMode.WORLD;
		params.Transform[3] = position;
		return params;
	}
}
