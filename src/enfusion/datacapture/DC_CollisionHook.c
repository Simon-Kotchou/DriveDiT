// ============================================================================
// DC_CollisionHook - forwards vanilla vehicle contacts to the dc-2.0 crash watch (DD-4-min)
// ============================================================================
//
// THE FIRST MODDED CLASS OF THE ADDON. It overrides one vanilla method of every vehicle in every world while the
// addon is loaded, so it follows three rules:
//   - super.ProcessCollision runs first and unchanged: the damage to the vehicle and its occupants, the instigator,
//     the side multiplier and the ejections stay exactly vanilla's;
//   - then DC_CrashWatch.NoteContact gets the owner and the container. It returns at once unless a dc-2.0 session is
//     capturing and watches this vehicle; otherwise it copies numbers and references into a preallocated record
//     (no allocation, no string);
//   - nothing is ever written to the container, the vehicle or the world.
//
// Vanilla 1.8.0.13 (Game/Components/Damage/SCR_VehicleDamageManagerComponent.c) calls ProcessCollision (:1510; :1511
// in the 1.9 index) from OnFrame (:1466-1506) once per new (vehicle, other entity) contact of the frame, skipping a
// contact that continues from the previous frame (:1489), and only for contacts that passed FilterContact (impulse
// >= MIN_IMPULSE_THRESHOLD 5, :1331-1334; none once the vehicle is destroyed, SCR_DamageManagerComponent.c:177-180)
// and OnFilteredContact (authority only, impulse >= m_fMinImpulse = mass x 2, :1343-1352, :619). OnWaterEnter calls it
// too, with the terrain as the other entity (:1288-1303). Signature, identical in the 1.8.0.13 and 1.9 indexes:
// void ProcessCollision(notnull SCR_CollisionDamageContainer collision). The container fields read by the crash watch
// (m_Other, m_fImpulse, m_fTotalMomentumOwner, GetOwnerVelocityBefore()) are SCR_CollisionDamageContainer.c:3-14, :39.
// The 1.9 method bodies (gates, continued-contact filter) are not extracted yet: re-check them at the 1.9 L-gate.
// ============================================================================

modded class SCR_VehicleDamageManagerComponent
{
	//------------------------------------------------------------------------------------------------
	//! Vanilla first, unchanged; then the contact goes to the capturing session's crash watch (a no-op otherwise)
	override void ProcessCollision(notnull SCR_CollisionDamageContainer collision)
	{
		super.ProcessCollision(collision);
		DC_CrashWatch.NoteContact(GetOwner(), collision);
	}
}
