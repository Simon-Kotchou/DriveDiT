// ============================================================================
// DC_CrashWatch - collision, damage, fire, driver, seat and rollover events of every session vehicle (DD-4-min)
// ============================================================================
//
// A DC_SessionModule (dispatch order 15: after DC_EgoSession 10, before DC_StreamRecorder 20) on the capture game
// mode. It writes the crash events of docs/capture_contract/dc-2.0.md section 9.5.1 into the events stream through
// DC_Session.LogVehicleEventInEpisode, each naming its vehicle slot (the ego now; traffic slots from DD-4/T2 call
// WatchSlot) and the episode in which it occurred.
//
// COLLISION DETECTOR (every watched slot, in every physics callback, OnStepPost). Rows are frame-bound (dc-2.0 freeze
//   decision D-A): one callback covers 1 to 3 merged physics steps of 1/30 s on 1.8 (probe run 4, E-10), so every
//   window is world time, never a count of steps or callbacks. data/dc2/crash_dv.py is the executable specification:
//   - each callback stores (world_time_ms, v_h) in a ring of RING_CAPACITY samples per slot, v_h the horizontal
//     (world x, z) part of Physics.GetVelocity() (Physics.c:99). The world frame: a spin turns the local vector
//     without any force;
//   - dv_h = |v_h - v_ref|, v_ref = the velocity at t - m_iCollisionWindowMs, interpolated between the newest sample
//     at or before that instant and the next one. Samples around it more than window + m_iCollisionWindowTolMs apart
//     (a hitch; a 3-step callback spans 100 ms) restart the history at the current sample and are not evaluated; a
//     history that does not reach back is not evaluated either;
//   - onset: dv_h >= m_fCollisionDvMps outside the refractory and the suppression (1000 ms after the slot is watched,
//     500 ms after a reposition); dv_mps is the peak of |v_h - v_ref| over m_iCollisionPeakMs of world time. The
//     4.0 m/s threshold is PROVISIONAL (TBD-10): it must be re-measured by the TBD-10 probe, per build_key and on
//     1.9. telemetry.dv_max_mps (TakeDvMax) carries the same dv_h, so a new value can be applied offline.
//
// CONTACTS (m_bUseContactHook; session.json events.contact_source)
//   The modded SCR_VehicleDamageManagerComponent.ProcessCollision (DC_CollisionHook.c) calls NoteContact once per new
//   (vehicle, other entity) contact that passed vanilla's filters. NoteContact returns at once for a vehicle that is
//   not watched; otherwise it copies numbers and references (and the step, episode and world time of the report)
//   into one of CONTACT_RECORDS preallocated records of the slot: no allocation, no string, no throw path (an
//   exception after vanilla's super would abort vanilla's OnFrame loop). Contacts within
//   [onset - 100 ms, onset + 250 ms] are kept until the pending row is written: the largest impulse names other, the
//   contacts with that entity merge into the row (src dv_contact), every other one becomes its own row (src contact,
//   never terminal), unless the slot's last row, within the refractory, had the same other entity (or none). A
//   pending row is written once a contact is in its window and the peak window passed, else once m_iEpisodeEndWaitMs
//   and 2 render frames passed. Without a contact another slot with an onset within 3 steps and 10 m names
//   other=vehicle and other_slot. other is classified on the contact's entity, then on each parent up to the root.
//
// POLLED STATE (every m_iStatePollMs of world time, OnStepPost; proto getters and cached hit zones only)
//   veh_damage: DamageManagerComponent.GetState() changes, or GetMovementDamage() crosses 1.0 (DamageManagerComponent.c
//   :56, :40); veh_fire_start / veh_fire_end: any cached SCR_FlammableHitZone is on fire for
//   SCR_DamageManagerComponent.IsOnFire(hz) (SCR_DamageManagerComponent.c:1036; no fire invoker exists);
//   driver_state: CharacterControllerComponent.GetLifeState() of the pilot-seat occupant (Vehicle.GetPilot,
//   Vehicle.c:114) or of the last one after the seat emptied; dismount / remount: the pilot seat empties / is
//   occupied again. The first poll writes a row only where the vehicle differs from the default (undamaged, not
//   burning, a seated driver alive). rollover: up_y below m_fRolloverUpY for m_iRolloverHoldMs of world time,
//   written when confirmed with onset_step = the first tilted step; re-armed after 2 s above 0.8.
//
// EPISODE OWNERSHIP: a row belongs to the episode of its occurrence (collision onset, contact report, first tilted
//   step, poll), snapshotted when it occurs, even when it is written after that episode ended; event_seq and
//   world_time_ms stay in write order. Step and episode are one pair: the last completed step
//   (DC_Session.GetLastPostStep) and the episode that owns it (GetLastStepEpisodeId), also for a report between
//   callbacks right after EndEpisode cleared the current episode.
//
// EGO TERMINATION (contract 9.5.1 / 9.6): a terminal ego event latches DC_StreamRecorder.LatchTermination when it is
//   confirmed (collision at the onset; veh_damage not undamaged or immobile, veh_fire_start: collision; rollover at
//   the end of the hold), only into its own episode and before the recorder's driver latch of that callback.
//   DC_EgoSession holds the episode end while HoldEpisodeEnd (the ego's collision row is pending, at most
//   m_iEpisodeEndWaitMs after the onset), refuses a new episode while IsEpisodeBlocked (a pending row, the
//   m_iCrashCooldownMs cool-down after the last terminal ego event, tilted, immobile, burning; the last three
//   "disabled" for at most m_iEpisodeBlockMaxMs, then the session ends with ego_disabled), and calls
//   FinishDestroyedSlot before it ends the session with ego_destroyed. The core is the only source of the
//   terminations collision and rollover (DC_StreamRecorder ignores them from a driver).
//
// REPOSITIONING: whoever moves a slot by SetWorldTransform calls PrepareReposition first (DC_DriverContext.
//   PrepareReposition for the ego): the pending row is written with its peak so far, the held contacts are written,
//   the history is cleared and the suppression starts before the move. The teleport or respawn row then calls
//   NoteRepositioned, which clears the history again.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! One vanilla contact of a watched vehicle (DC_CrashWatch.NoteContact), kept until a row uses it. Preallocated per
//! slot: the hook only copies numbers and references into a free record
class DC_CrashContact
{
	bool m_bHeld;				//!< holds a contact that no row has used yet
	int m_iWorldMs;				//!< world_time_ms when ProcessCollision reported it
	int m_iStep;				//!< the last completed physics_step then (the onset_step of a contact-only row)
	int m_iEpisode = -1;		//!< the episode then (the row's episode_id)
	IEntity m_Other;			//!< the classified entity of the other side (weak)
	int m_iOtherKind;			//!< DC_CrashWatch.OTHER_*
	int m_iOtherSlot = -1;		//!< slot of the other vehicle when it is watched
	float m_fImpulse;			//!< SCR_CollisionDamageContainer.m_fImpulse [N s]
	float m_fDv;				//!< m_fTotalMomentumOwner / mass [m/s]
	float m_fSpeedBefore;		//!< horizontal speed before the contact [m/s]
}

//------------------------------------------------------------------------------------------------
//! The watch state of one vehicle slot
class DC_CrashSlot
{
	int m_iSlotId;
	IEntity m_Vehicle;
	Vehicle m_Car;								//!< m_Vehicle as Vehicle (pilot seat); null for another entity type
	SCR_DamageManagerComponent m_Damage;
	ref array<SCR_FlammableHitZone> m_aFlammable = {};
	ref array<ref DC_CrashContact> m_aContacts = {};
	bool m_bFinished;							//!< FinishDestroyedSlot ran

	// velocity history: a ring of (world_time_ms, v_h); m_iRingHead is the newest sample
	ref array<int> m_aSampleMs = {};
	ref array<float> m_aVelX = {};
	ref array<float> m_aVelZ = {};
	int m_iRingHead = -1;
	int m_iRingCount;
	float m_fDvMax;								//!< largest dv_h since the last TakeDvMax
	bool m_bHasDvMax;
	int m_iSuppressUntilMs;
	int m_iRefractoryUntilMs;

	// the pending collision row (onset seen, row not written yet)
	bool m_bPending;
	int m_iOnsetStep;
	int m_iOnsetMs;
	int m_iOnsetFrame;
	int m_iOnsetEpisode = -1;					//!< the episode of the onset (the row's episode_id)
	vector m_vOnsetPos;
	float m_fRefX;								//!< v_ref of the onset
	float m_fRefZ;
	float m_fPeak;								//!< peak |v_h(t) - v_ref|
	float m_fPeakX;
	float m_fPeakZ;
	float m_fFwdX;								//!< horizontal forward and right axes at the onset, normalised
	float m_fFwdZ;
	float m_fRightX;
	float m_fRightZ;

	// the last written collision row (merge rule) and the last onset (registry match)
	bool m_bHasLastRow;
	int m_iLastRowMs;
	IEntity m_LastRowOther;
	bool m_bHasLastOnset;
	int m_iLastOnsetStep;
	vector m_vLastOnsetPos;

	// rollover
	bool m_bRollArmed = true;
	bool m_bTilted;
	int m_iTiltSinceMs;
	int m_iTiltSinceStep;						//!< the rollover's onset_step
	int m_iTiltEpisode = -1;
	bool m_bUpright;
	int m_iUprightSinceMs;

	// polled state (the defaults of contract 9.5.1)
	EDamageState m_eDamageState = EDamageState.UNDAMAGED;
	bool m_bImmobile;
	bool m_bOnFire;
	IEntity m_Driver;							//!< the pilot-seat occupant, or the last one after the seat emptied
	bool m_bHasDriver;
	ECharacterLifeState m_eDriverState = ECharacterLifeState.ALIVE;
	bool m_bInSeat;
	bool m_bDismounted;
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 crash events (contract 9.5.1): world-time delta-v collisions, vanilla contacts, damage, fire, driver, seat and rollover of every session vehicle; the ego's crash termination and cool-down")]
class DC_CrashWatchClass : DC_SessionModuleClass
{
}

//------------------------------------------------------------------------------------------------
class DC_CrashWatch : DC_SessionModule
{
	// === DETECTOR (session.json events.*, TBD-10; PROVISIONAL until the TBD-10 probe, re-measured per build_key and on 1.9) ===
	[Attribute("4", UIWidgets.Slider, "Collision onset: horizontal world-frame delta-v over the window of world time [m/s] (TBD-10, PROVISIONAL: 2 x the E-01 full-braking maximum per 100 ms; re-measure per build_key and on 1.9)", "0.5 50 0.1", category: "Crash")]
	protected float m_fCollisionDvMps;

	[Attribute("100", UIWidgets.Slider, "Delta-v window [ms of world time]: v_ref is interpolated at t - window (TBD-10)", "10 1000 1", category: "Crash")]
	protected int m_iCollisionWindowMs;

	[Attribute("50", UIWidgets.Slider, "Gap tolerance [ms]: samples around t - window more than window + this apart restart the history (TBD-10)", "0 1000 1", category: "Crash")]
	protected int m_iCollisionWindowTolMs;

	[Attribute("200", UIWidgets.Slider, "dv_mps is the peak delta-v over this much world time after the onset [ms] (TBD-10)", "0 2000 1", category: "Crash")]
	protected int m_iCollisionPeakMs;

	[Attribute("1000", UIWidgets.Slider, "Per-slot refractory after an onset [ms]: further crossings belong to the same impact (TBD-10)", "0 10000 10", category: "Crash")]
	protected int m_iCollisionRefractoryMs;

	[Attribute("250", UIWidgets.Slider, "An episode end waits at most this long after the onset for the ego's collision row [ms] (TBD-10)", "0 2000 1", category: "Crash")]
	protected int m_iEpisodeEndWaitMs;

	[Attribute("3000", UIWidgets.Slider, "No new episode until this long after the ego's last terminal crash event [ms] (TBD-10: refractory + rollover hold + a poll + the row wait, and the car comes to rest)", "0 60000 100", category: "Crash")]
	protected int m_iCrashCooldownMs;

	[Attribute("60000", UIWidgets.Slider, "A session whose ego stays tilted, immobile or burning between episodes this long (not destroyed) ends with ego_disabled [ms] (TBD-10)", "1000 600000 1000", category: "Crash")]
	protected int m_iEpisodeBlockMaxMs;

	[Attribute("0.5", UIWidgets.Slider, "rollover when up_y stays below this (0.5 = 60 deg of tilt; TBD-10)", "-1 1 0.01", category: "Crash")]
	protected float m_fRolloverUpY;

	[Attribute("500", UIWidgets.Slider, "rollover when up_y stays below rollover_up_y this long [ms] (TBD-10)", "0 10000 10", category: "Crash")]
	protected int m_iRolloverHoldMs;

	[Attribute("100", UIWidgets.Slider, "Damage, fire, driver and seat poll period [ms of world time] (TBD-10)", "10 1000 10", category: "Crash")]
	protected int m_iStatePollMs;

	[Attribute("0", UIWidgets.CheckBox, "Use the contacts the modded SCR_VehicleDamageManagerComponent.ProcessCollision forwards (events.contact_source vanilla_process_collision); off: none", category: "Crash")]
	protected bool m_bUseContactHook;

	// === CONSTANTS (contract 9.5.1) ===
	static const int RING_CAPACITY = 32;			//!< samples per slot: the window at any step of 5 ms or more
	static const int SUPPRESS_AFTER_WATCH_MS = 1000;
	static const int SUPPRESS_AFTER_TELEPORT_MS = 500;
	static const int CONTACT_LEAD_MS = 100;			//!< a contact may come this long before its onset
	static const int CONTACT_LAG_MS = 250;			//!< ... or this long after it
	static const int CONTACT_RECORDS = 4;
	static const int ROW_MIN_FRAMES = 2;
	static const int PARTNER_STEPS = 3;
	static const float PARTNER_DIST_M = 10;
	static const int PARENT_DEPTH_MAX = 8;			//!< classification walks at most this many parents
	static const float ROLLOVER_REARM_UP_Y = 0.8;
	static const int ROLLOVER_REARM_MS = 2000;
	static const string TERMINATION_COLLISION = "collision";
	static const string TERMINATION_ROLLOVER = "rollover";
	static const string CONTACT_SOURCE_HOOK = "vanilla_process_collision";
	static const string CONTACT_SOURCE_NONE = "none";
	static const string LOG = "[DC_Crash] ";

	//! collision payload other (contract 9.5.1), indexed by OTHER_*
	static const int OTHER_UNKNOWN = 0;
	static const int OTHER_VEHICLE = 1;
	static const int OTHER_CHARACTER = 2;
	static const int OTHER_TREE = 3;
	static const int OTHER_BUILDING = 4;
	static const int OTHER_ROCK = 5;
	static const int OTHER_TERRAIN = 6;
	static const int OTHER_STATIC = 7;
	static const int OTHER_DYNAMIC = 8;
	protected static ref array<string> s_aOtherNames = {"unknown", "vehicle", "character", "tree", "building", "rock", "terrain", "static", "dynamic"};

	//! The watch of the capturing session: the target of the vanilla hook (DC_CollisionHook.c), null otherwise
	protected static DC_CrashWatch s_Active;

	// === STATE ===
	protected DC_Session m_Session;
	protected DC_StreamRecorder m_Recorder;
	protected ref array<ref DC_CrashSlot> m_aSlots = {};
	protected bool m_bOpen;
	protected int m_iNextPollMs;
	protected bool m_bEgoTerminalSeen;
	protected int m_iLastEgoTerminalMs;			//!< world_time_ms of the ego's last terminal crash event

	// session_end.json observed
	protected int m_iCollisionRows;
	protected int m_iContactRows;
	protected int m_iContactsSeen;
	protected int m_iContactsMerged;
	protected int m_iContactsOverwritten;
	protected int m_iHistoryRestarts;

	//------------------------------------------------------------------------------------------------
	override int GetDispatchOrder()
	{
		return 15;
	}

	//------------------------------------------------------------------------------------------------
	override bool OnSessionPrepare(notnull DC_Session session)
	{
		if (SCR_Global.IsEditMode())
			return false;

		m_Session = session;
		m_Recorder = DC_StreamRecorder.Cast(GetOwner().FindComponent(DC_StreamRecorder));
		if (!m_Recorder)
		{
			session.RefuseReadiness("no DC_StreamRecorder on the game mode (the crash watch latches the ego's termination there)");
			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! session.json events.* (contract 10.1, in its order)
	override void WriteSessionJson(string section, notnull DC_JsonWriter json)
	{
		if (section == "events")
		{
			json.AddFloat("collision_dv_mps", m_fCollisionDvMps, 3);
			json.AddInt("collision_window_ms", m_iCollisionWindowMs);
			json.AddInt("collision_window_tol_ms", m_iCollisionWindowTolMs);
			json.AddInt("collision_peak_ms", m_iCollisionPeakMs);
			json.AddInt("collision_refractory_ms", m_iCollisionRefractoryMs);
			json.AddInt("episode_end_wait_ms", m_iEpisodeEndWaitMs);
			json.AddInt("crash_cooldown_ms", m_iCrashCooldownMs);
			json.AddInt("episode_block_max_ms", m_iEpisodeBlockMaxMs);
			json.AddFloat("rollover_up_y", m_fRolloverUpY, 3);
			json.AddInt("rollover_hold_ms", m_iRolloverHoldMs);
			json.AddInt("state_poll_ms", m_iStatePollMs);
			json.AddString("contact_source", GetContactSource());
			return;
		}

		if (section == "observed")
			json.AddBool("crash_threshold_provisional", true);
	}

	//------------------------------------------------------------------------------------------------
	//! The ego is watched from the open on (traffic slots from DD-4/T2 call WatchSlot); the vanilla hook targets this
	override bool OnSessionOpen(notnull DC_Session session)
	{
		m_iNextPollMs = 0;
		m_bOpen = true;
		s_Active = this;
		IEntity ego = session.GetEgo();
		if (ego)
			WatchSlot(session.GetEgoSlotId(), ego);

		Print(string.Format(LOG + "watching %1 slot(s): collision dv %2 m/s over %3 ms of world time (gap tolerance %4 ms, PROVISIONAL), contacts %5", m_aSlots.Count(), m_fCollisionDvMps.ToString(-1, 2), m_iCollisionWindowMs, m_iCollisionWindowTolMs, GetContactSource()), LogLevel.NORMAL);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Delta-v detector and rollover per slot, then the state poll every m_iStatePollMs
	override void OnStepPost(int physicsStep, float timeSlice, int worldMs)
	{
		if (!m_bOpen)
			return;

		foreach (DC_CrashSlot slot : m_aSlots)
		{
			StepSlot(slot, physicsStep, worldMs);
		}

		if (worldMs < m_iNextPollMs)
			return;

		int period = m_iStatePollMs;
		if (period < 1)
			period = 1;

		m_iNextPollMs = worldMs - worldMs % period + period;
		foreach (DC_CrashSlot polled : m_aSlots)
		{
			PollSlot(polled);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Pending collision rows and unused contacts are written here (the contacts arrive in the vanilla OnFrame)
	override void OnFrame(int renderFrame, float timeSlice, int worldMs)
	{
		if (!m_bOpen)
			return;

		foreach (DC_CrashSlot slot : m_aSlots)
		{
			FlushSlot(slot, worldMs, renderFrame, false);
		}
	}

	//------------------------------------------------------------------------------------------------
	override bool IsDrained()
	{
		foreach (DC_CrashSlot slot : m_aSlots)
		{
			if (slot.m_bPending)
				return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Every pending row is written before the streams are committed (DC_EgoSession already flushed before it ended
	//! the episode); the counters go to session_end.json observed
	override void OnSessionClosing(string endReason)
	{
		FlushPendingRows();
		m_Session.NoteObserved("crash_collision_rows", m_iCollisionRows.ToString());
		m_Session.NoteObserved("crash_contact_rows", m_iContactRows.ToString());
		m_Session.NoteObserved("crash_contacts_seen", m_iContactsSeen.ToString());
		m_Session.NoteObserved("crash_contacts_merged", m_iContactsMerged.ToString());
		m_Session.NoteObserved("crash_contacts_overwritten", m_iContactsOverwritten.ToString());
		m_Session.NoteObserved("crash_history_restarts", m_iHistoryRestarts.ToString());
		m_bOpen = false;
	}

	//------------------------------------------------------------------------------------------------
	override void OnSessionRelease(bool teardown)
	{
		m_bOpen = false;
		if (s_Active == this)
			s_Active = null;

		m_aSlots.Clear();
		m_Recorder = null;
	}

	//------------------------------------------------------------------------------------------------
	//! World teardown calls no module: the hook must not reach this component any more
	override void OnDelete(IEntity owner)
	{
		m_bOpen = false;
		if (s_Active == this)
			s_Active = null;

		super.OnDelete(owner);
	}

	//------------------------------------------------------------------------------------------------
	// Module API (DC_EgoSession, DC_StreamRecorder, DC_Session, DC_DriverContext, traffic from DD-4/T2)
	//------------------------------------------------------------------------------------------------
	//! Watch a vehicle slot from now on (its slot id from DC_VehicleRegistry). Writes no row: the first poll compares
	//! with the defaults of contract 9.5.1. A slot already watched is left as it is
	void WatchSlot(int slotId, notnull IEntity vehicle)
	{
		if (FindSlot(slotId))
			return;

		DC_CrashSlot slot = new DC_CrashSlot();
		slot.m_iSlotId = slotId;
		slot.m_Vehicle = vehicle;
		slot.m_Car = Vehicle.Cast(vehicle);
		slot.m_Damage = SCR_DamageManagerComponent.GetDamageManager(vehicle);
		if (slot.m_Damage)
		{
			array<HitZone> zones = {};
			slot.m_Damage.GetAllHitZones(zones);
			foreach (HitZone zone : zones)
			{
				SCR_FlammableHitZone flammable = SCR_FlammableHitZone.Cast(zone);
				if (flammable)
					slot.m_aFlammable.Insert(flammable);
			}
		}

		slot.m_aSampleMs.Resize(RING_CAPACITY);
		slot.m_aVelX.Resize(RING_CAPACITY);
		slot.m_aVelZ.Resize(RING_CAPACITY);
		for (int i = 0; i < CONTACT_RECORDS; i++)
		{
			slot.m_aContacts.Insert(new DC_CrashContact());
		}

		slot.m_iSuppressUntilMs = m_Session.GetWorldMs() + SUPPRESS_AFTER_WATCH_MS;
		if (slot.m_Car)
		{
			IEntity pilot = slot.m_Car.GetPilot();
			if (pilot)
			{
				slot.m_Driver = pilot;
				slot.m_bHasDriver = true;
				slot.m_bInSeat = true;
			}
		}

		m_aSlots.Insert(slot);
		string damageText = "no damage manager";
		int flammableCount = slot.m_aFlammable.Count();
		if (slot.m_Damage)
			damageText = flammableCount.ToString() + " flammable hit zone(s)";

		Print(string.Format(LOG + "slot %1 watched (%2, pilot seat %3)", slotId, damageText, DC_LaunchGuard.BoolText(slot.m_bInSeat)), LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! telemetry.dv_max_mps: the largest dv_h of the slot since the last call. false while none was evaluated
	bool TakeDvMax(int slotId, out float value)
	{
		value = 0;
		DC_CrashSlot slot = FindSlot(slotId);
		if (!slot || !slot.m_bHasDvMax)
			return false;

		value = slot.m_fDvMax;
		slot.m_bHasDvMax = false;
		slot.m_fDvMax = 0;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! true while the slot's collision row is pending and younger than m_iEpisodeEndWaitMs: the episode end waits (the
	//! row then carries the episode id and precedes episode_end). Past the wait the row is written now
	bool HoldEpisodeEnd(int slotId)
	{
		DC_CrashSlot slot = FindSlot(slotId);
		if (!m_bOpen || !slot || !slot.m_bPending)
			return false;

		int worldMs = m_Session.GetWorldMs();
		if (worldMs < slot.m_iOnsetMs + m_iEpisodeEndWaitMs)
			return true;

		FlushSlot(slot, worldMs, m_Session.GetRenderFrame(), true);
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! true while no new episode may begin for the slot (contract 9.5.1): its collision row is pending, the ego's last
	//! terminal crash event is younger than m_iCrashCooldownMs, or the vehicle is tilted, immobile or burning
	//! \param[out] reason the first blocking condition, empty when none
	//! \param[out] disabled true when that condition is tilted, immobile or burning (unbounded by itself; the caller
	//! ends the session after GetEpisodeBlockMaxMs of it)
	bool IsEpisodeBlocked(int slotId, out string reason, out bool disabled)
	{
		reason = string.Empty;
		disabled = false;
		DC_CrashSlot slot = FindSlot(slotId);
		if (!slot)
			return false;

		if (slot.m_bPending)
			reason = "its collision row is pending";
		else if (slotId == m_Session.GetEgoSlotId() && m_bEgoTerminalSeen && m_Session.GetWorldMs() < m_iLastEgoTerminalMs + m_iCrashCooldownMs)
			reason = "the crash cool-down runs";
		else if (slot.m_bTilted)
			reason = "tilted";
		else if (slot.m_Damage && slot.m_Damage.GetMovementDamage() >= 1)
			reason = "immobile";
		else if (FireStateOf(slot) != SCR_EBurningState.NONE)
			reason = "burning";

		disabled = reason == "tilted" || reason == "immobile" || reason == "burning";
		return !reason.IsEmpty();
	}

	//------------------------------------------------------------------------------------------------
	//! session.json events.episode_block_max_ms: how long a disabled ego may block episodes
	int GetEpisodeBlockMaxMs()
	{
		return m_iEpisodeBlockMaxMs;
	}

	//------------------------------------------------------------------------------------------------
	//! The slot's vehicle is destroyed and the session ends now: its pending collision row and contacts are written,
	//! then its state is polled once more (veh_damage state=destroyed, veh_fire_start when it burns). Idempotent
	void FinishDestroyedSlot(int slotId)
	{
		DC_CrashSlot slot = FindSlot(slotId);
		if (!m_bOpen || !slot || slot.m_bFinished)
			return;

		slot.m_bFinished = true;
		FlushSlot(slot, m_Session.GetWorldMs(), m_Session.GetRenderFrame(), true);
		PollSlot(slot);
	}

	//------------------------------------------------------------------------------------------------
	//! Every pending row and every unused contact of every slot is written now
	void FlushPendingRows()
	{
		if (!m_bOpen)
			return;

		int worldMs = m_Session.GetWorldMs();
		int renderFrame = m_Session.GetRenderFrame();
		foreach (DC_CrashSlot slot : m_aSlots)
		{
			FlushSlot(slot, worldMs, renderFrame, true);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Before a slot is moved by SetWorldTransform (a teleport or respawn; contract 9.5.1): its pending row is written
	//! with the peak so far and its held contacts are written, then its history is cleared and the suppression starts,
	//! so neither the jump nor the post-reset velocity is compared with the pre-reset history
	static void PrepareReposition(int slotId)
	{
		if (s_Active)
			s_Active.IsolateSlot(slotId);
	}

	//------------------------------------------------------------------------------------------------
	//! A teleport or respawn row of the slot was logged (DC_Session.LogVehicleEventInEpisode): the same isolation
	//! again from now, also for a reset that did not call PrepareReposition
	static void NoteRepositioned(int slotId)
	{
		if (s_Active)
			s_Active.IsolateSlot(slotId);
	}

	//------------------------------------------------------------------------------------------------
	//! One vanilla contact of a vehicle (the modded ProcessCollision, after vanilla ran): no-op unless the vehicle is
	//! watched by the capturing session. Copies numbers and references only
	static void NoteContact(IEntity owner, notnull SCR_CollisionDamageContainer collision)
	{
		if (s_Active && owner)
			s_Active.AddContact(owner, collision);
	}

	//------------------------------------------------------------------------------------------------
	//! See PrepareReposition and NoteRepositioned: the pending row (peak so far) and the held contacts are written, the
	//! history is cleared, the tilt timer restarts and the suppression runs from now
	void IsolateSlot(int slotId)
	{
		DC_CrashSlot slot = FindSlot(slotId);
		if (!m_bOpen || !slot)
			return;

		int worldMs = m_Session.GetWorldMs();
		FlushSlot(slot, worldMs, m_Session.GetRenderFrame(), true);

		slot.m_iRingCount = 0;
		slot.m_bTilted = false;
		slot.m_bUpright = false;
		int until = worldMs + SUPPRESS_AFTER_TELEPORT_MS;
		if (until > slot.m_iSuppressUntilMs)
			slot.m_iSuppressUntilMs = until;
	}

	//------------------------------------------------------------------------------------------------
	//! See NoteContact. Exception-free: every reference is checked before use
	void AddContact(notnull IEntity owner, notnull SCR_CollisionDamageContainer collision)
	{
		if (!m_bOpen || !m_bUseContactHook)
			return;

		DC_CrashSlot slot = FindSlotByEntity(owner);
		if (!slot)
			return;

		m_iContactsSeen++;
		int otherSlot = -1;
		IEntity named;
		int otherKind = ClassifyOther(collision.m_Other, otherSlot, named);
		DC_CrashContact record = FreeRecord(slot);
		if (!record)
			return;

		record.m_bHeld = true;
		record.m_iWorldMs = m_Session.GetWorldMs();
		record.m_iStep = m_Session.GetLastPostStep();
		record.m_iEpisode = m_Session.GetLastStepEpisodeId();		// the pair (last completed step, its episode)
		record.m_Other = named;
		record.m_iOtherKind = otherKind;
		record.m_iOtherSlot = otherSlot;
		record.m_fImpulse = collision.m_fImpulse;
		record.m_fDv = 0;
		Physics physics = owner.GetPhysics();
		if (physics && physics.GetMass() > 0)
			record.m_fDv = collision.m_fTotalMomentumOwner / physics.GetMass();

		vector before = collision.GetOwnerVelocityBefore();
		record.m_fSpeedBefore = Math.Sqrt(before[0] * before[0] + before[2] * before[2]);
	}

	//------------------------------------------------------------------------------------------------
	// Detector
	//------------------------------------------------------------------------------------------------
	protected void StepSlot(notnull DC_CrashSlot slot, int physicsStep, int worldMs)
	{
		IEntity vehicle = slot.m_Vehicle;
		if (!vehicle || vehicle.IsDeleted())
		{
			slot.m_iRingCount = 0;
			return;
		}

		Physics physics = vehicle.GetPhysics();
		if (!physics)
		{
			slot.m_iRingCount = 0;
			return;
		}

		vector velocity = physics.GetVelocity();
		vector transform[4];
		vehicle.GetWorldTransform(transform);
		vector upAxis = transform[1];
		StepRollover(slot, upAxis[1], physicsStep, worldMs);

		// the peak of a pending onset, against its own v_ref, over m_iCollisionPeakMs of world time
		if (slot.m_bPending && worldMs <= slot.m_iOnsetMs + m_iCollisionPeakMs)
		{
			float px = velocity[0] - slot.m_fRefX;
			float pz = velocity[2] - slot.m_fRefZ;
			float peak = Math.Sqrt(px * px + pz * pz);
			if (peak > slot.m_fPeak)
			{
				slot.m_fPeak = peak;
				slot.m_fPeakX = px;
				slot.m_fPeakZ = pz;
			}
		}

		float refX;
		float refZ;
		if (!SampleVelocity(slot, worldMs, velocity[0], velocity[2], refX, refZ))
			return;

		float dx = velocity[0] - refX;
		float dz = velocity[2] - refZ;
		float dv = Math.Sqrt(dx * dx + dz * dz);
		if (!slot.m_bHasDvMax || dv > slot.m_fDvMax)
		{
			slot.m_fDvMax = dv;
			slot.m_bHasDvMax = true;
		}

		if (dv < m_fCollisionDvMps || slot.m_bPending || worldMs < slot.m_iSuppressUntilMs || worldMs < slot.m_iRefractoryUntilMs)
			return;

		// Onset: v_ref and the episode are kept; the axes are the vehicle's at the onset (dv_lon forward-positive)
		slot.m_bPending = true;
		slot.m_iOnsetStep = physicsStep;
		slot.m_iOnsetMs = worldMs;
		slot.m_iOnsetFrame = m_Session.GetRenderFrame();
		slot.m_iOnsetEpisode = m_Session.GetLastStepEpisodeId();
		slot.m_vOnsetPos = transform[3];
		slot.m_fRefX = refX;
		slot.m_fRefZ = refZ;
		slot.m_fPeak = dv;
		slot.m_fPeakX = dx;
		slot.m_fPeakZ = dz;
		vector fwd = transform[2];
		vector right = transform[0];
		float fwdLength = Math.Sqrt(fwd[0] * fwd[0] + fwd[2] * fwd[2]);
		float rightLength = Math.Sqrt(right[0] * right[0] + right[2] * right[2]);
		if (fwdLength < 0.001)
			fwdLength = 1;

		if (rightLength < 0.001)
			rightLength = 1;

		slot.m_fFwdX = fwd[0] / fwdLength;
		slot.m_fFwdZ = fwd[2] / fwdLength;
		slot.m_fRightX = right[0] / rightLength;
		slot.m_fRightZ = right[2] / rightLength;
		slot.m_iRefractoryUntilMs = worldMs + m_iCollisionRefractoryMs;
		NoteTerminal(slot, TERMINATION_COLLISION, slot.m_iOnsetEpisode, worldMs);
	}

	//------------------------------------------------------------------------------------------------
	//! Adds the sample (worldMs, vx, vz) to the slot's ring and interpolates v_ref at worldMs - window (contract
	//! 9.5.1, data/dc2/crash_dv.py). A sample whose world time does not advance replaces the newest one.
	//! \return false when the callback is not evaluated (no reference yet, or a gap: the history then restarts)
	protected bool SampleVelocity(notnull DC_CrashSlot slot, int worldMs, float vx, float vz, out float refX, out float refZ)
	{
		refX = 0;
		refZ = 0;
		if (slot.m_iRingCount > 0 && worldMs <= slot.m_aSampleMs.Get(slot.m_iRingHead))
		{
			slot.m_aVelX.Set(slot.m_iRingHead, vx);
			slot.m_aVelZ.Set(slot.m_iRingHead, vz);
			return false;
		}

		slot.m_iRingHead = (slot.m_iRingHead + 1) % RING_CAPACITY;
		slot.m_aSampleMs.Set(slot.m_iRingHead, worldMs);
		slot.m_aVelX.Set(slot.m_iRingHead, vx);
		slot.m_aVelZ.Set(slot.m_iRingHead, vz);
		if (slot.m_iRingCount < RING_CAPACITY)
			slot.m_iRingCount++;

		int target = worldMs - m_iCollisionWindowMs;
		for (int back = 1; back < slot.m_iRingCount; back++)
		{
			int older = (slot.m_iRingHead - back + RING_CAPACITY) % RING_CAPACITY;
			int olderMs = slot.m_aSampleMs.Get(older);
			if (olderMs > target)
				continue;

			int newer = (older + 1) % RING_CAPACITY;
			int newerMs = slot.m_aSampleMs.Get(newer);
			if (newerMs - olderMs > m_iCollisionWindowMs + m_iCollisionWindowTolMs)
			{
				// a gap (a hitch, a reset): the history restarts at the current sample, nothing is compared across it
				slot.m_iRingCount = 1;
				m_iHistoryRestarts++;
				return false;
			}

			float part = target - olderMs;
			float span = newerMs - olderMs;
			float f = 0;
			if (span > 0)
				f = part / span;

			float olderX = slot.m_aVelX.Get(older);
			float olderZ = slot.m_aVelZ.Get(older);
			refX = olderX + (slot.m_aVelX.Get(newer) - olderX) * f;
			refZ = olderZ + (slot.m_aVelZ.Get(newer) - olderZ) * f;
			return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! up_y below m_fRolloverUpY for m_iRolloverHoldMs: rollover with onset_step = the first tilted step (once until
	//! re-armed by 2 s above 0.8)
	protected void StepRollover(notnull DC_CrashSlot slot, float upY, int physicsStep, int worldMs)
	{
		if (upY < m_fRolloverUpY)
		{
			slot.m_bUpright = false;
			if (!slot.m_bTilted)
			{
				slot.m_bTilted = true;
				slot.m_iTiltSinceMs = worldMs;
				slot.m_iTiltSinceStep = physicsStep;
				slot.m_iTiltEpisode = m_Session.GetLastStepEpisodeId();
			}

			if (slot.m_bRollArmed && worldMs - slot.m_iTiltSinceMs >= m_iRolloverHoldMs)
			{
				slot.m_bRollArmed = false;
				LogRowIn(slot, slot.m_iTiltEpisode, "rollover", "onset_step=" + slot.m_iTiltSinceStep.ToString() + ";up_y=" + FormatFloat(upY));
				NoteTerminal(slot, TERMINATION_ROLLOVER, slot.m_iTiltEpisode, worldMs);
			}

			return;
		}

		slot.m_bTilted = false;
		if (upY <= ROLLOVER_REARM_UP_Y)
		{
			slot.m_bUpright = false;
			return;
		}

		if (!slot.m_bUpright)
		{
			slot.m_bUpright = true;
			slot.m_iUprightSinceMs = worldMs;
		}

		if (!slot.m_bRollArmed && worldMs - slot.m_iUprightSinceMs >= ROLLOVER_REARM_MS)
			slot.m_bRollArmed = true;
	}

	//------------------------------------------------------------------------------------------------
	// Rows
	//------------------------------------------------------------------------------------------------
	//! The pending row once it is complete (or at once with force), then the contacts no onset used
	protected void FlushSlot(notnull DC_CrashSlot slot, int worldMs, int renderFrame, bool force)
	{
		if (slot.m_bPending)
		{
			bool peakDone = worldMs >= slot.m_iOnsetMs + m_iCollisionPeakMs;
			bool waited = (worldMs >= slot.m_iOnsetMs + m_iEpisodeEndWaitMs) && (renderFrame - slot.m_iOnsetFrame >= ROW_MIN_FRAMES);
			if (force || waited || (peakDone && HasWindowContact(slot)))
				WriteCollisionRow(slot);
		}

		// Contacts wait for a pending row (it may use them) and for CONTACT_LEAD_MS (an onset may still follow)
		if (slot.m_bPending)
			return;

		foreach (DC_CrashContact record : slot.m_aContacts)
		{
			if (!record.m_bHeld || (!force && worldMs < record.m_iWorldMs + CONTACT_LEAD_MS))
				continue;

			record.m_bHeld = false;
			if (MergesWithLastRow(slot, record))
			{
				m_iContactsMerged++;
				continue;
			}

			WriteContactRow(slot, record);
		}
	}

	//------------------------------------------------------------------------------------------------
	protected bool HasWindowContact(notnull DC_CrashSlot slot)
	{
		foreach (DC_CrashContact record : slot.m_aContacts)
		{
			if (record.m_bHeld && IsInOnsetWindow(slot, record))
				return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------------------------
	protected bool IsInOnsetWindow(notnull DC_CrashSlot slot, notnull DC_CrashContact record)
	{
		return (record.m_iWorldMs >= slot.m_iOnsetMs - CONTACT_LEAD_MS) && (record.m_iWorldMs <= slot.m_iOnsetMs + CONTACT_LAG_MS);
	}

	//------------------------------------------------------------------------------------------------
	//! A contact without an onset is the impact of the slot's last row when it came within the refractory and named
	//! the same other entity (or the last row named none)
	protected bool MergesWithLastRow(notnull DC_CrashSlot slot, notnull DC_CrashContact record)
	{
		if (!slot.m_bHasLastRow || record.m_iWorldMs - slot.m_iLastRowMs >= m_iCollisionRefractoryMs)
			return false;

		if (!slot.m_LastRowOther)
			return true;

		return record.m_Other == slot.m_LastRowOther;
	}

	//------------------------------------------------------------------------------------------------
	//! The collision row of the pending onset (contract 9.5.1 key order), into the onset's episode. The contacts in
	//! the onset window are chosen now: the largest impulse names other and the contacts with that entity merge; the
	//! others stay held and become their own rows
	protected void WriteCollisionRow(notnull DC_CrashSlot slot)
	{
		DC_CrashContact best;
		foreach (DC_CrashContact record : slot.m_aContacts)
		{
			if (!record.m_bHeld || !IsInOnsetWindow(slot, record))
				continue;

			if (!best || record.m_fImpulse > best.m_fImpulse)
				best = record;
		}

		int otherKind = OTHER_UNKNOWN;
		int otherSlot = -1;
		IEntity other;
		float impulse;
		string src = "dv";
		if (best)
		{
			otherKind = best.m_iOtherKind;
			otherSlot = best.m_iOtherSlot;
			other = best.m_Other;
			impulse = best.m_fImpulse;
			src = "dv_contact";
			foreach (DC_CrashContact merged : slot.m_aContacts)
			{
				if (merged.m_bHeld && IsInOnsetWindow(slot, merged) && merged.m_Other == other)
				{
					merged.m_bHeld = false;
					m_iContactsMerged++;
				}
			}
		}
		else
		{
			otherSlot = FindPartnerSlot(slot);
			if (otherSlot >= 0)
				otherKind = OTHER_VEHICLE;
		}

		float dvLon = slot.m_fPeakX * slot.m_fFwdX + slot.m_fPeakZ * slot.m_fFwdZ;
		float dvLat = slot.m_fPeakX * slot.m_fRightX + slot.m_fPeakZ * slot.m_fRightZ;
		float speedBefore = Math.Sqrt(slot.m_fRefX * slot.m_fRefX + slot.m_fRefZ * slot.m_fRefZ);
		string payload = "onset_step=" + slot.m_iOnsetStep.ToString() + ";dv_mps=" + FormatFloat(slot.m_fPeak) + ";dv_lon_mps=" + FormatFloat(dvLon) + ";dv_lat_mps=" + FormatFloat(dvLat) + ";v_mps=" + FormatFloat(speedBefore);
		payload += OtherKeys(otherKind, otherSlot, other);
		if (best)
			payload += ";impulse_ns=" + FormatFloat(impulse);

		payload += ";src=" + src;
		slot.m_bPending = false;
		slot.m_bHasLastRow = true;
		slot.m_iLastRowMs = slot.m_iOnsetMs;
		slot.m_LastRowOther = other;
		slot.m_bHasLastOnset = true;
		slot.m_iLastOnsetStep = slot.m_iOnsetStep;
		slot.m_vLastOnsetPos = slot.m_vOnsetPos;
		m_iCollisionRows++;
		LogRowIn(slot, slot.m_iOnsetEpisode, "collision", payload);
	}

	//------------------------------------------------------------------------------------------------
	//! A collision row of a contact without an onset (src contact: below the detector, never terminal), at the step and
	//! in the episode of its report
	protected void WriteContactRow(notnull DC_CrashSlot slot, notnull DC_CrashContact record)
	{
		string payload = "onset_step=" + record.m_iStep.ToString() + ";dv_mps=" + FormatFloat(record.m_fDv) + ";v_mps=" + FormatFloat(record.m_fSpeedBefore);
		payload += OtherKeys(record.m_iOtherKind, record.m_iOtherSlot, record.m_Other);
		payload += ";impulse_ns=" + FormatFloat(record.m_fImpulse) + ";src=contact";
		slot.m_bHasLastRow = true;
		slot.m_iLastRowMs = record.m_iWorldMs;
		slot.m_LastRowOther = record.m_Other;
		m_iContactRows++;
		LogRowIn(slot, record.m_iEpisode, "collision", payload);
	}

	//------------------------------------------------------------------------------------------------
	//! Another watched slot with an onset within PARTNER_STEPS steps and PARTNER_DIST_M of this one's, else -1
	protected int FindPartnerSlot(notnull DC_CrashSlot slot)
	{
		foreach (DC_CrashSlot candidate : m_aSlots)
		{
			if (candidate == slot)
				continue;

			int step;
			vector position;
			if (candidate.m_bPending)
			{
				step = candidate.m_iOnsetStep;
				position = candidate.m_vOnsetPos;
			}
			else if (candidate.m_bHasLastOnset)
			{
				step = candidate.m_iLastOnsetStep;
				position = candidate.m_vLastOnsetPos;
			}
			else
			{
				continue;
			}

			if (Math.AbsInt(step - slot.m_iOnsetStep) <= PARTNER_STEPS && vector.Distance(position, slot.m_vOnsetPos) <= PARTNER_DIST_M)
				return candidate.m_iSlotId;
		}

		return -1;
	}

	//------------------------------------------------------------------------------------------------
	//! ";other=<kind>" and ";other_slot=<n>" (a watched vehicle) or ";other_prefab=<path>" (contract 9.5.1)
	protected string OtherKeys(int otherKind, int otherSlot, IEntity other)
	{
		string text = ";other=" + s_aOtherNames[otherKind];
		if (otherSlot >= 0)
			return text + ";other_slot=" + otherSlot.ToString();

		if (!other)
			return text;

		string prefab = PrefabPath(other);
		if (!prefab.IsEmpty())
			text += ";other_prefab=" + prefab;

		return text;
	}

	//------------------------------------------------------------------------------------------------
	// Polled state
	//------------------------------------------------------------------------------------------------
	protected void PollSlot(notnull DC_CrashSlot slot)
	{
		IEntity vehicle = slot.m_Vehicle;
		if (!vehicle || vehicle.IsDeleted())
			return;

		// a poll inside OnStepPost or between callbacks (FinishDestroyedSlot): the last completed step and its episode
		int worldMs = m_Session.GetWorldMs();
		int episodeId = m_Session.GetLastStepEpisodeId();
		SCR_DamageManagerComponent damage = slot.m_Damage;
		if (damage)
		{
			EDamageState damageState = damage.GetState();
			float movementDamage = damage.GetMovementDamage();
			bool immobile = movementDamage >= 1;
			if (damageState != slot.m_eDamageState || immobile != slot.m_bImmobile)
			{
				slot.m_eDamageState = damageState;
				slot.m_bImmobile = immobile;
				LogRowIn(slot, episodeId, "veh_damage", "state=" + DamageStateName(damageState) + ";health=" + FormatFloat(damage.GetHealthScaled()) + ";movement_damage=" + FormatFloat(movementDamage));
				if (damageState != EDamageState.UNDAMAGED || immobile)
					NoteTerminal(slot, TERMINATION_COLLISION, episodeId, worldMs);
			}

			SCR_EBurningState fireState = FireStateOf(slot);
			bool onFire = fireState != SCR_EBurningState.NONE;
			if (onFire != slot.m_bOnFire)
			{
				slot.m_bOnFire = onFire;
				if (onFire)
				{
					LogRowIn(slot, episodeId, "veh_fire_start", "fire_state=" + FireStateName(fireState));
					NoteTerminal(slot, TERMINATION_COLLISION, episodeId, worldMs);
				}
				else
				{
					LogRowIn(slot, episodeId, "veh_fire_end", string.Empty);
				}
			}
		}

		PollDriver(slot, vehicle, episodeId);
	}

	//------------------------------------------------------------------------------------------------
	//! driver_state, dismount and remount of the pilot seat (contract 9.5.1)
	protected void PollDriver(notnull DC_CrashSlot slot, notnull IEntity vehicle, int episodeId)
	{
		if (!slot.m_Car)
			return;

		IEntity pilot = slot.m_Car.GetPilot();
		bool inSeat;
		if (pilot)
		{
			inSeat = true;
			slot.m_Driver = pilot;
			slot.m_bHasDriver = true;
		}

		if (slot.m_bHasDriver)
		{
			ECharacterLifeState lifeState = LifeStateOf(slot.m_Driver, slot.m_eDriverState);
			if (lifeState != slot.m_eDriverState)
			{
				slot.m_eDriverState = lifeState;
				LogRowIn(slot, episodeId, "driver_state", "state=" + LifeStateName(lifeState) + ";in_seat=" + DC_LaunchGuard.BoolText(inSeat));
			}
		}

		if (slot.m_bInSeat && !inSeat)
		{
			slot.m_bInSeat = false;
			slot.m_bDismounted = true;
			LogRowIn(slot, episodeId, "dismount", "driver=" + LifeStateName(slot.m_eDriverState) + ";v_mps=" + FormatFloat(HorizontalSpeedOf(vehicle)));
			return;
		}

		if (!slot.m_bInSeat && inSeat)
		{
			// The first occupancy of a seat that was empty when the slot was watched writes no row (contract 9.5.1)
			slot.m_bInSeat = true;
			if (slot.m_bDismounted)
				LogRowIn(slot, episodeId, "remount", "driver=" + LifeStateName(slot.m_eDriverState));
		}
	}

	//------------------------------------------------------------------------------------------------
	//! The highest fire state over the flammable hit zones while any of them is on fire (IsOnFire respects the
	//! prefab's minimum burning state), else NONE
	protected SCR_EBurningState FireStateOf(notnull DC_CrashSlot slot)
	{
		SCR_EBurningState highest = SCR_EBurningState.NONE;
		if (!slot.m_Damage)
			return highest;

		bool burning;
		foreach (SCR_FlammableHitZone zone : slot.m_aFlammable)
		{
			if (!zone)
				continue;

			SCR_EBurningState zoneState = zone.GetFireState();
			if (zoneState > highest)
				highest = zoneState;

			if (zoneState != SCR_EBurningState.NONE && slot.m_Damage.IsOnFire(zone))
				burning = true;
		}

		if (!burning)
			return SCR_EBurningState.NONE;

		return highest;
	}

	//------------------------------------------------------------------------------------------------
	// Helpers
	//------------------------------------------------------------------------------------------------
	//! A terminal crash event of a slot confirmed now (contract 9.5.1): for the ego it starts the cool-down and is
	//! latched as the termination of its own episode only (DC_StreamRecorder keeps the first one)
	protected void NoteTerminal(notnull DC_CrashSlot slot, string reason, int episodeId, int worldMs)
	{
		if (slot.m_iSlotId != m_Session.GetEgoSlotId())
			return;

		m_bEgoTerminalSeen = true;
		m_iLastEgoTerminalMs = worldMs;
		if (m_Recorder && episodeId >= 0 && episodeId == m_Session.GetEpisodeId())
			m_Recorder.LatchTermination(reason);
	}

	//------------------------------------------------------------------------------------------------
	//! One events row of the slot, owned by episodeId (the episode of its occurrence)
	protected void LogRowIn(notnull DC_CrashSlot slot, int episodeId, string kind, string payload)
	{
		m_Session.LogVehicleEventInEpisode(slot.m_iSlotId, episodeId, kind, payload);
	}

	//------------------------------------------------------------------------------------------------
	protected DC_CrashSlot FindSlot(int slotId)
	{
		foreach (DC_CrashSlot slot : m_aSlots)
		{
			if (slot.m_iSlotId == slotId)
				return slot;
		}

		return null;
	}

	//------------------------------------------------------------------------------------------------
	//! The watched slot of an entity by reference (no string, no allocation: the hook path)
	protected DC_CrashSlot FindSlotByEntity(IEntity entity)
	{
		if (!entity)
			return null;

		foreach (DC_CrashSlot slot : m_aSlots)
		{
			if (slot.m_Vehicle == entity)
				return slot;
		}

		return null;
	}

	//------------------------------------------------------------------------------------------------
	//! A record for a new contact: a free one, else the oldest is overwritten (counted)
	protected DC_CrashContact FreeRecord(notnull DC_CrashSlot slot)
	{
		DC_CrashContact oldest;
		foreach (DC_CrashContact record : slot.m_aContacts)
		{
			if (!record.m_bHeld)
				return record;

			if (!oldest || record.m_iWorldMs < oldest.m_iWorldMs)
				oldest = record;
		}

		m_iContactsOverwritten++;
		return oldest;
	}

	//------------------------------------------------------------------------------------------------
	//! OTHER_* of the contact's other entity: the classes of contract 9.5.1 are tested on the entity, then on each
	//! parent up to the root (a generator tree or a composition part is its own class); without a match the entity's
	//! physics decides dynamic or static
	//! \param[out] otherSlot the watched slot of a vehicle, else -1
	//! \param[out] named the entity that matched (the other entity itself without a match): identity and prefab
	protected int ClassifyOther(IEntity other, out int otherSlot, out IEntity named)
	{
		otherSlot = -1;
		named = other;
		if (!other)
			return OTHER_UNKNOWN;

		IEntity level = other;
		for (int depth = 0; level && depth < PARENT_DEPTH_MAX; depth++)
		{
			int levelKind = ClassifyEntity(level, otherSlot);
			if (levelKind != OTHER_UNKNOWN)
			{
				named = level;
				return levelKind;
			}

			level = level.GetParent();
		}

		Physics physics = other.GetPhysics();
		if (!physics)
			return OTHER_UNKNOWN;

		if (physics.IsDynamic())
			return OTHER_DYNAMIC;

		return OTHER_STATIC;
	}

	//------------------------------------------------------------------------------------------------
	//! OTHER_* of one entity by the class order of contract 9.5.1, OTHER_UNKNOWN when none matches
	//! \param[out] otherSlot the watched slot of the entity, else -1 (always assigned)
	protected int ClassifyEntity(notnull IEntity entity, out int otherSlot)
	{
		otherSlot = -1;
		DC_CrashSlot watched = FindSlotByEntity(entity);
		if (watched)
		{
			otherSlot = watched.m_iSlotId;
			return OTHER_VEHICLE;
		}

		if (Vehicle.Cast(entity))
			return OTHER_VEHICLE;

		if (ChimeraCharacter.Cast(entity))
			return OTHER_CHARACTER;

		if (BaseTree.Cast(entity))
			return OTHER_TREE;

		if (Building.Cast(entity))
			return OTHER_BUILDING;

		if (SCR_IndestructibleEnvironmentalEntity.Cast(entity))
			return OTHER_ROCK;

		if (GenericTerrainEntity.Cast(entity))
			return OTHER_TERRAIN;

		return OTHER_UNKNOWN;
	}

	//------------------------------------------------------------------------------------------------
	protected string GetContactSource()
	{
		if (m_bUseContactHook)
			return CONTACT_SOURCE_HOOK;

		return CONTACT_SOURCE_NONE;
	}

	//------------------------------------------------------------------------------------------------
	//! Horizontal speed of an entity's physics body [m/s], 0 without one
	static float HorizontalSpeedOf(notnull IEntity entity)
	{
		Physics physics = entity.GetPhysics();
		if (!physics)
			return 0;

		vector velocity = physics.GetVelocity();
		return Math.Sqrt(velocity[0] * velocity[0] + velocity[2] * velocity[2]);
	}

	//------------------------------------------------------------------------------------------------
	//! The life state of a character entity, fallback when it is gone or has no controller
	static ECharacterLifeState LifeStateOf(IEntity entity, ECharacterLifeState fallback)
	{
		ChimeraCharacter character = ChimeraCharacter.Cast(entity);
		if (!character)
			return fallback;

		CharacterControllerComponent controller = character.GetCharacterController();
		if (!controller)
			return fallback;

		return controller.GetLifeState();
	}

	//------------------------------------------------------------------------------------------------
	//! The prefab of an entity without its {GUID} prefix, safe as a payload value (no ',', '=', ';', '"', blanks)
	static string PrefabPath(notnull IEntity entity)
	{
		string path = DC_VehicleRegistry.PrefabOf(entity);
		int close = path.IndexOf("}");
		if (path.StartsWith("{") && close >= 0)
			path = path.Substring(close + 1, path.Length() - close - 1);

		path.Replace("=", "_");
		return DC_ProbeCsv.CleanField(path);
	}

	//------------------------------------------------------------------------------------------------
	//! Payload floats: 2 decimals (contract 9.5.1)
	static string FormatFloat(float value)
	{
		return value.ToString(-1, 2);
	}

	//------------------------------------------------------------------------------------------------
	//! veh_damage state: EDamageState STATE1 to STATE3 are written intermediary (contract 9.5.1)
	static string DamageStateName(EDamageState damageState)
	{
		if (damageState == EDamageState.UNDAMAGED)
			return "undamaged";

		if (damageState == EDamageState.DESTROYED)
			return "destroyed";

		return "intermediary";
	}

	//------------------------------------------------------------------------------------------------
	static string LifeStateName(ECharacterLifeState lifeState)
	{
		if (lifeState == ECharacterLifeState.DEAD)
			return "dead";

		if (lifeState == ECharacterLifeState.INCAPACITATED)
			return "incapacitated";

		return "alive";
	}

	//------------------------------------------------------------------------------------------------
	//! veh_fire_start fire_state of an SCR_EBurningState above NONE
	static string FireStateName(SCR_EBurningState fireState)
	{
		if (fireState >= SCR_EBurningState.BURNING)
			return "burning";

		if (fireState >= SCR_EBurningState.SMOKING_IGNITING)
			return "smoking_igniting";

		if (fireState >= SCR_EBurningState.SMOKING_HEAVY)
			return "smoking_heavy";

		return "smoking_light";
	}
}
