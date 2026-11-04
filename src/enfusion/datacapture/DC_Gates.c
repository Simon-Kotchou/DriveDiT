// ============================================================================
// DC_Gates - opens every vehicle gate of a world (bar gates, wide metal gates)
// ============================================================================
//
// A closed gate on a car's route makes the vanilla driver stop, get out, open it and get back in
// (HandleFailedNavlinkDoor.bt), because an AI agent below max LOD never opens a gate remotely
// (SCR_AISelectDoorOperatorAgent.c:91-94). A car under direct control (an expert driver) runs into it.
// DC_Gates.OpenVehicleGates opens them all once; CheckOpen reports, some seconds later, every leaf that is not open.
//
//   DC_Gates gates = DC_Gates.OpenVehicleGates(GetGame().GetWorld(), "[DataCapture]");	// once, authority only
//   ...
//   gates.CheckOpen(10);	// e.g. 10 s later, from the caller's own timer
//
// Used by SCR_MLDataCaptureDirector at its start (m_bOpenGatesAtStart); any other session start can call the same.
// The gates stay open: an opened gate let every later AI car through in the 2026-11-03 Play (DD-1
// run2/DISMOUNT_SOURCE.md 2.3). Every API is checked against the 1.8.0.13 scripts (file:line below) and is the same
// in 1.9.0.10.
// ============================================================================

class DC_Gates : Managed
{
	//! Vehicle navmesh project of SCR_AIWorld.et: a door leaf with a link of it is a gate cars wait at
	//! (SCR_AIWorld.HasVehicleNavlink, Game/AI/SCR_AIWorld.c:315-321)
	static const string VEHICLE_NAVMESH = "BTRlike";
	//! Door control value set: fully open, the SCR_DoorOpenerEntity default (Game/Campaign/SCR_doorOpener.c:24)
	static const float OPEN_CONTROL_VALUE = 1.0;
	//! A door is open beyond 1 minus this of its normalized state (SCR_AIOpenDoor.STATE_EPSILON, SCR_AIOpenDoor.c:4,39)
	static const float OPEN_EPSILON = 0.01;

	protected string m_sLogPrefix;
	protected ref array<IEntity> m_aLeaves = {};	//!< door leaves with a vehicle navmesh link (the query)
	protected ref array<IEntity> m_aDoors = {};		//!< every door leaf to open: the gate leaves and their sibling leaves
	protected ref array<IEntity> m_aGates = {};		//!< the gates: the leaves' parents (door set or barrier)
	protected int m_iDoorsSeen;						//!< entities with a DoorComponent the query saw

	//! Counts of OpenVehicleGates
	protected int m_iWithSmartAction;
	protected int m_iOpening;
	protected int m_iAlreadyOpen;
	protected int m_iNoDoorComponent;
	protected int m_iQueryMs;

	//------------------------------------------------------------------------------------------------
	//! Opens every vehicle gate of the world once and logs every gate with its position and a summary with the counts.
	//! Call it on the authority, once the world is loaded.
	//! \param[in] world the world to open the gates of (its whole bound box)
	//! \param[in] logPrefix put in front of every log line, e.g. "[DataCapture]"
	//! \return the opener: the counts (GetGateCount, GetDoorCount, GetOpeningCount) and the door leaves for CheckOpen
	static DC_Gates OpenVehicleGates(notnull BaseWorld world, string logPrefix)
	{
		DC_Gates opener = new DC_Gates(logPrefix);
		opener.OpenAll(world);
		return opener;
	}

	//------------------------------------------------------------------------------------------------
	void DC_Gates(string logPrefix)
	{
		m_sLogPrefix = logPrefix;
	}

	//------------------------------------------------------------------------------------------------
	//! \return the number of vehicle gates found
	int GetGateCount()
	{
		return m_aGates.Count();
	}

	//------------------------------------------------------------------------------------------------
	//! \return the number of door leaves of those gates
	int GetDoorCount()
	{
		return m_aDoors.Count();
	}

	//------------------------------------------------------------------------------------------------
	//! \return the number of door leaves that were closed and were set open
	int GetOpeningCount()
	{
		return m_iOpening;
	}

	//------------------------------------------------------------------------------------------------
	//! 1. One query over the world's bound box (BaseWorld.GetBoundBox / QueryEntitiesByAABB,
	//!    Core/generated/World/BaseWorld.c:116, :56) keeps every entity with a DoorComponent and a navmesh link of the
	//!    vehicle navmesh (AddLeaf). The OpenGate smart action alone would not do: every door leaf inherits it from
	//!    Door_Base.et, house doors included, while only gate leaves add a BTRlike link
	//!    (Door_MetalGate_01_Long_LEFT_EXT_ST_B.et, BarGate_01_bar_base.et).
	//! 2. Each leaf's sibling leaves are added: the doors the vanilla driver opens (SA_OpenGate.bt runs SCR_AIOpenDoor
	//!    with m_bOpenSiblingDoors 1; SCR_AIOpenDoor.FindDoorActions, Game/AI/ScriptedNodes/Soldier/SCR_AIOpenDoor.c:55).
	//! 3. Each closed leaf is set open with BaseDoorComponent.SetControlValue(1, RplId.Invalid())
	//!    (Game/generated/Components/BaseDoorComponent.c:27), as vanilla SCR_DoorOpenerEntity opens the doors around it
	//!    without any character (Game/Campaign/SCR_doorOpener.c:54-71). The user action the AI uses,
	//!    SCR_DoorUserAction.PerformAction, needs a performer: it calls the native UseDoorAction(user), which toggles
	//!    the door and opens it relative to that user (SCR_DoorUserAction.c:1-14, BaseDoorComponent.c:35), and vanilla
	//!    only ever passes the operating character (SCR_AIOpenDoor.c:24-26, :43).
	//! One-time cost: one native query visiting every entity of the map with one script callback each, logged in ms.
	protected void OpenAll(BaseWorld world)
	{
		vector mins;
		vector maxs;
		world.GetBoundBox(mins, maxs);

		int startMs = System.GetTickCount(); // Core/generated/System/System.c:134
		world.QueryEntitiesByAABB(mins, maxs, AddLeaf);
		m_iQueryMs = System.GetTickCount() - startMs;

		array<ScriptedUserAction> actions = {};
		foreach (IEntity leaf : m_aLeaves)
		{
			// The tag the vanilla gate reaction looks for (SCR_AIGoalReaction.c:474; SCR_AIUtils.c:502)
			if (SCR_AISmartActionHandling.FindSmartAction(leaf, "OpenGate", false))
				m_iWithSmartAction++;

			IEntity leafGate = GetGateOf(leaf);
			if (!m_aGates.Contains(leafGate))
				m_aGates.Insert(leafGate);

			SCR_AIOpenDoor.FindDoorActions(leaf, actions, true);
			foreach (ScriptedUserAction action : actions)
			{
				IEntity actionDoor = action.GetOwner(); // Game/generated/UserAction/BaseUserAction.c:76
				if (actionDoor && !m_aDoors.Contains(actionDoor))
					m_aDoors.Insert(actionDoor);
			}

			if (!m_aDoors.Contains(leaf))
				m_aDoors.Insert(leaf);
		}

		foreach (int gateIndex, IEntity gate : m_aGates)
		{
			// Initialised explicitly: declared in the loop body, they count one gate each.
			int gateLeaves = 0;
			int gateOpening = 0;
			int gateAlreadyOpen = 0;
			foreach (IEntity door : m_aDoors)
			{
				if (GetGateOf(door) != gate)
					continue;

				gateLeaves++;
				DoorComponent doorComponent = DoorComponent.Cast(door.FindComponent(DoorComponent));
				if (!doorComponent)
				{
					m_iNoDoorComponent++;
					continue;
				}

				if (IsDoorOpen(doorComponent) || doorComponent.IsOpening()) // BaseDoorComponent.c:23
				{
					gateAlreadyOpen++;
					continue;
				}

				doorComponent.SetControlValue(OPEN_CONTROL_VALUE, RplId.Invalid()); // RplId.Invalid: Core/generated/Replication/RplId.c:18
				gateOpening++;
			}

			m_iOpening += gateOpening;
			m_iAlreadyOpen += gateAlreadyOpen;
			Print(string.Format("%1 Gate %2/%3: %4 at %5 - %6 door leaves, opening %7, already open %8", m_sLogPrefix, gateIndex + 1, m_aGates.Count(), DescribePrefab(gate), gate.GetOrigin(), gateLeaves, gateOpening, gateAlreadyOpen), LogLevel.NORMAL);
		}

		string counts = string.Format("%1 vehicle gates with %2 door leaves (%3 leaves with a vehicle navmesh link, %4 of them with the OpenGate smart action, among %5 doors of the map)", m_aGates.Count(), m_aDoors.Count(), m_aLeaves.Count(), m_iWithSmartAction, m_iDoorsSeen);
		LogLevel level = LogLevel.NORMAL;
		if (m_aGates.IsEmpty() || m_iNoDoorComponent > 0)
			level = LogLevel.WARNING;

		Print(string.Format("%1 Gates: found %2 in %3 ms over the map %4 - %5; opening %6 leaves, %7 already open, %8 without a door component", m_sLogPrefix, counts, m_iQueryMs, mins, maxs, m_iOpening, m_iAlreadyOpen, m_iNoDoorComponent), level);
	}

	//------------------------------------------------------------------------------------------------
	//! Counts the door leaves that are open and logs every one that is not (blocked while it opened, or destroyed) with
	//! its position; a closed one is set open once more. Call it once, some seconds after OpenVehicleGates
	//! (DoorSet_MetalGate_01_Long opens in 1.5 s).
	//! \param[in] afterS seconds since OpenVehicleGates, for the log lines
	//! \return the number of door leaves that are not open or are gone
	int CheckOpen(int afterS)
	{
		int open;
		int notOpen;
		int gone;
		foreach (IEntity door : m_aDoors)
		{
			if (!door)
			{
				gone++;
				continue;
			}

			DoorComponent doorComponent = DoorComponent.Cast(door.FindComponent(DoorComponent));
			if (!doorComponent)
			{
				gone++;
				continue;
			}

			if (IsDoorOpen(doorComponent))
			{
				open++;
				continue;
			}

			notOpen++;
			float doorState = doorComponent.GetNormalizedDoorState(); // Game/generated/Components/BaseDoorComponent.c:17
			float controlValue = doorComponent.GetControlValue(); // BaseDoorComponent.c:29
			IEntity gate = GetGateOf(door);
			Print(string.Format("%1 WARNING: gate door %2 at %3 (gate %4 at %5) is not open %6 s after it was opened: state %7, control value %8, opening %9 - set open again; a car may still stop there", m_sLogPrefix, DescribePrefab(door), door.GetOrigin(), DescribePrefab(gate), gate.GetOrigin(), afterS, doorState.ToString(-1, 2), controlValue.ToString(-1, 2), doorComponent.IsOpening()), LogLevel.WARNING);
			doorComponent.SetControlValue(OPEN_CONTROL_VALUE, RplId.Invalid());
		}

		LogLevel level = LogLevel.NORMAL;
		if (notOpen > 0 || gone > 0)
			level = LogLevel.WARNING;

		Print(string.Format("%1 Gates %2 s after opening: %3 of %4 door leaves open, %5 not open, %6 gone (destroyed or deleted)", m_sLogPrefix, afterS, open, m_aDoors.Count(), notOpen, gone), level);
		return notOpen + gone;
	}

	//------------------------------------------------------------------------------------------------
	//! QueryEntitiesCallback (Core/proto/EnWorld.c:6-7) of the gate query: keeps every door leaf with a navmesh link of
	//! the vehicle navmesh. \return true: go on with the query
	protected bool AddLeaf(IEntity entity)
	{
		if (!entity || !entity.FindComponent(DoorComponent))
			return true;

		m_iDoorsSeen++;
		if (HasVehicleDoorLink(entity) && !m_aLeaves.Contains(entity))
			m_aLeaves.Insert(entity);

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! \return true when a navmesh custom link of the entity is for the vehicle navmesh: SCR_AIWorld.HasVehicleNavlink
	//! (Game/AI/SCR_AIWorld.c:315-321), but over every NavmeshCustomLinkComponent, since a gate leaf carries a door link
	//! and a jump link (Door_MetalGate_01_Long_LEFT_EXT_ST_B.et)
	protected bool HasVehicleDoorLink(IEntity entity)
	{
		array<Managed> navlinks = {};
		entity.FindComponents(NavmeshCustomLinkComponent, navlinks); // Core/generated/Entities/IEntity.c:539
		foreach (Managed navlinkComponent : navlinks)
		{
			NavmeshCustomLinkComponent navlink = NavmeshCustomLinkComponent.Cast(navlinkComponent);
			if (navlink && navlink.HasLinkOfNavmeshType(VEHICLE_NAVMESH)) // Game/generated/AI/NavmeshCustomLinkComponent.c:14
				return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! \return the gate a door leaf belongs to: its parent (the door set or barrier), or the leaf itself
	protected IEntity GetGateOf(IEntity door)
	{
		IEntity parent = door.GetParent(); // Core/generated/Entities/IEntity.c:158
		if (parent)
			return parent;

		return door;
	}

	//------------------------------------------------------------------------------------------------
	//! \return true when the door is fully open, the test of SCR_AIOpenDoor (SCR_AIOpenDoor.c:39)
	protected bool IsDoorOpen(DoorComponent doorComponent)
	{
		return Math.AbsFloat(doorComponent.GetNormalizedDoorState()) > 1.0 - OPEN_EPSILON;
	}

	//------------------------------------------------------------------------------------------------
	//! \return the entity's prefab path (EntityPrefabData.GetPrefabName, Core/generated/Entities/EntityPrefabData.c:20),
	//! "no prefab" or "none"
	protected string DescribePrefab(IEntity entity)
	{
		if (!entity)
			return "none";

		EntityPrefabData prefabData = entity.GetPrefabData(); // Core/generated/Entities/IEntity.c:181
		if (!prefabData)
			return "no prefab";

		ResourceName prefab = prefabData.GetPrefabName();
		if (prefab.IsEmpty())
			return "no prefab";

		return prefab.GetPath(); // Core/generated/Types/ResourceName.c:14
	}
}
