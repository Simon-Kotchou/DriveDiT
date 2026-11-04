// ============================================================================
// DC_VehicleRegistry - vehicle slot identity of one dc-2.0 session (DD-4-min, r2 3.9, C35)
// ============================================================================
//
// One registry per session (owned by DC_EgoSession; never static, so a second Play without a script reload starts
// from slot 0 again and nothing leaks between sessions):
//   - Register gives a vehicle the next slot_id; slot ids are monotonic and never reused within the session (a
//     registered entity keeps its slot, a released slot is never given out again);
//   - the slot is keyed to the entity's IEntity.GetID() as an opaque ASCII token (EntityID.ToString(), the vanilla
//     form of SCR_GameModeStatistics.c:417) and to its prefab, in map-keyed caches (slot -> token, token -> slot,
//     slot -> prefab), so nothing is ever looked up by entity name;
//   - CheckIdentity asserts at capture time that the entity seen in a slot still carries the registered token: a
//     replaced or re-created entity is reported, never silently renamed (C35).
// T0-MIN registers the ego only (slot 0); traffic slots come with DD-4 (T2).
// ============================================================================

//------------------------------------------------------------------------------------------------
class DC_VehicleRegistry
{
	protected int m_iNextSlotId;
	protected ref map<int, string> m_mEntityIdBySlot = new map<int, string>();
	protected ref map<string, int> m_mSlotByEntityId = new map<string, int>();
	protected ref map<int, string> m_mPrefabBySlot = new map<int, string>();
	protected ref array<int> m_aReleased = {};

	//------------------------------------------------------------------------------------------------
	//! \return the entity's slot: its existing one, else the next slot id (monotonic, never reused)
	int Register(notnull IEntity entity)
	{
		string entityId = EntityIdOf(entity);
		int slotId;
		if (m_mSlotByEntityId.Find(entityId, slotId))
			return slotId;

		slotId = m_iNextSlotId;
		m_iNextSlotId++;
		m_mEntityIdBySlot.Insert(slotId, entityId);
		m_mSlotByEntityId.Insert(entityId, slotId);
		m_mPrefabBySlot.Insert(slotId, PrefabOf(entity));
		return slotId;
	}

	//------------------------------------------------------------------------------------------------
	//! The slot leaves the session (its vehicle is gone); the id is never given out again
	void Release(int slotId)
	{
		if (m_aReleased.Find(slotId) < 0)
			m_aReleased.Insert(slotId);
	}

	//------------------------------------------------------------------------------------------------
	//! Asserts at capture time that entity is the vehicle registered in slotId (same GetID token, slot not released)
	//! \param[out] failure one line, empty when the identity holds
	bool CheckIdentity(int slotId, IEntity entity, out string failure)
	{
		failure = string.Empty;
		string registered;
		if (!m_mEntityIdBySlot.Find(slotId, registered))
		{
			failure = "slot " + slotId.ToString() + " was never registered";
			return false;
		}

		if (m_aReleased.Find(slotId) >= 0)
		{
			failure = "slot " + slotId.ToString() + " was released";
			return false;
		}

		if (!entity)
		{
			failure = "slot " + slotId.ToString() + " has no entity";
			return false;
		}

		string current = EntityIdOf(entity);
		if (current != registered)
		{
			failure = "slot " + slotId.ToString() + " was " + registered + ", the entity now is " + current;
			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The registered GetID token of a slot, empty for an unknown slot
	string GetEntityId(int slotId)
	{
		string entityId;
		m_mEntityIdBySlot.Find(slotId, entityId);
		return entityId;
	}

	//------------------------------------------------------------------------------------------------
	//! The registered prefab of a slot, empty for an unknown slot
	string GetPrefab(int slotId)
	{
		string prefab;
		m_mPrefabBySlot.Find(slotId, prefab);
		return prefab;
	}

	//------------------------------------------------------------------------------------------------
	//! Slots given out so far
	int GetSlotCount()
	{
		return m_iNextSlotId;
	}

	//------------------------------------------------------------------------------------------------
	//! IEntity.GetID() as an opaque ASCII token (EntityID.ToString(): "0x" and 16 hex digits)
	static string EntityIdOf(notnull IEntity entity)
	{
		EntityID entityId = entity.GetID();
		return entityId.ToString();
	}

	//------------------------------------------------------------------------------------------------
	//! Prefab resource name, empty for an entity without prefab data
	static string PrefabOf(notnull IEntity entity)
	{
		EntityPrefabData prefabData = entity.GetPrefabData();
		if (!prefabData)
			return string.Empty;

		return prefabData.GetPrefabName();
	}
}
