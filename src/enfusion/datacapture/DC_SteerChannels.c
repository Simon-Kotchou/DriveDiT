// ============================================================================
// DC_SteerChannels - raw wheel-angle candidates of a driverless car for CC-07 step 0 and CC-08 (DD-14b)
// ============================================================================
//
// The vehicle simulation has no per-wheel steer angle getter on either build (VehicleWheeledSimulation.c of 1.8.0.13
// and 1.9.0.37; G19_0_STATIC_1.9.0.37.md section 8). This class records every candidate that a script can read
// without a character, once per physics step (Read, from OnProbePostSimulate) and the rotator bones also once per
// render frame after its physics update (ReadPostFrame, from OnProbePostFrame), so CC-07 step 0 can validate one of
// them against the path curvature in the low-slip regime and measure its lag. None of them is a label yet
// (MIGRATION_1_9.md 8.3, DD-4-probe (d)): they are raw candidates, in rad, in the vehicle frame (x right, y up,
// z forward), positive = the wheel points to the right (clockwise seen from above, the sign of heading_deg and of a
// right-turn yaw rate; positive steering turns right on both builds, e02a steer_sign_used=1).
//
// CANDIDATES (columns)
//   wa_rot_l/r   yaw of the kingpin pivot bone <bone><rotator suffix> (S105: v_wheel_L01_rotator, the pivot of the
//                1.9 KingpinAxes block, S105_base.et:1009-1014 of 1.9.0.37; the right one is assumed mirrored), from
//                Animation.GetBoneMatrix (model space; Core/generated/Visual/Animation.c:16, unchanged 1.8 -> 1.9;
//                IEntity.GetAnimation, IEntity.c:512)
//   wa_bone_l/r  yaw of the wheel pivot bone <bone> (v_wheel_L01: the WheelPosition pivot and the wheel slot pivot)
//   wa_ent_l/r   yaw of the wheel entity attached to the slot (SlotManagerComponent.GetSlotByName,
//                EntitySlotInfo.GetAttachedEntity; S105 slots Wheel_L01 / Wheel_R01), world transform taken into the
//                vehicle frame with IEntity.VectorToLocal
//   sig_steer    the vehicle's own steering signal (S105 AnimVariables: variable "steering" <- signal "steering0",
//                S105_base.et:913-914, unchanged 1.8 -> 1.9) from the vehicle's SignalsManagerComponent
//                (Vehicle_Base.et; FindSignal / GetSignalValue, Game/generated/Components/SignalsManagerComponent.c:
//                41, 48). Unit unknown (input -1..1 or an angle): recorded raw
//   wa_rot_l/r_pf  the rotator yaw read at the last EOnPostFrame before this step's row (render-frame timing check)
//   slip_lat_l/r WheelGetLateralSlip of the two front wheels (unit not documented; low-slip gating cross-check)
//   vdir_l/r     direction of the front wheel's model-space velocity, atan2(v_x, v_z) of WheelGetVelocityLS (rad;
//                empty below 0.5 m/s): close to the wheel angle minus the slip angle in the low-slip regime
//
// WHEEL SPIN REMOVED. Each bone or entity source uses one axis of its own frame: the column of its matrix that points
// most along the vehicle's x axis at the first read (|x| share >= AXLE_MIN_SHARE, else the source is unusable and
// stays empty), signed to point to +x. The spin of a wheel turns its frame about that axle, so the axle and its yaw
// atan2(-a_z, a_x) do not change with the spin; the rotator bone carries no spin at all. The first read happens in
// the first reset of a probe (steering commanded 0), so the axle choice is made at zero command and the toe at zero
// command shows as the value itself. The choice and its x share go into the run header at the probe stop
// (steer.channel.<source>.*).
//
// TIMING: bones are animated in the frame update, so the value read after a physics step may be the previous frame's;
// the analysis measures the lag against the steering readback (CC-07 step 0: no lag beyond one physics step). The
// vehicle has to be animated for the bone and entity sources: the steering probes keep a chase camera on it
// (DC_ProbeSteerBase, DC_RigR1) so its animation is never culled by distance.
//
// BUILD-AGNOSTIC: every member exists with the same declaration in 1.8.0.13 and 1.9.0.37 (no overlay needed).
// ============================================================================

//------------------------------------------------------------------------------------------------
class DC_SteerChannels
{
	static const int SRC_ROT_L = 0;
	static const int SRC_ROT_R = 1;
	static const int SRC_BONE_L = 2;
	static const int SRC_BONE_R = 3;
	static const int SRC_ENT_L = 4;
	static const int SRC_ENT_R = 5;
	static const int SOURCE_COUNT = 6;

	//! An axis is the axle when this share of its length lies along the vehicle x axis at the first read
	static const float AXLE_MIN_SHARE = 0.9;
	//! Axis choice states (m_aAxis)
	static const int AXIS_NOT_CHOSEN = -1;
	static const int AXIS_UNUSABLE = -2;
	//! vdir_l/r stay empty below this wheel speed [m/s]
	static const float VDIR_MIN_SPEED = 0.5;

	protected IEntity m_Vehicle;
	protected VehicleWheeledSimulation m_Simulation;
	protected bool m_bAnimation;				//!< IEntity.GetAnimation was not null at InitChannels (fetched again per read)
	protected SignalsManagerComponent m_Signals;
	protected string m_sSignalName;
	protected int m_iSignal = -1;
	protected int m_iWheelL = -1;
	protected int m_iWheelR = -1;
	protected string m_sFailure;

	// Per source (SRC_*): name, bone index (bone sources) or attached entity (entity sources), axle choice and value
	protected ref array<string> m_aSourceNames = {};
	protected ref array<int> m_aBones = {};
	protected ref array<IEntity> m_aEntities = {};
	protected ref array<int> m_aAxis = {};
	protected ref array<float> m_aAxisSign = {};
	protected ref array<float> m_aAxisShare = {};
	protected ref array<float> m_aValue = {};
	protected ref array<bool> m_aValid = {};
	protected ref array<int> m_aReadsValid = {};

	protected float m_fSignal;
	protected bool m_bSignalValid;
	protected float m_fRotLPf;
	protected float m_fRotRPf;
	protected bool m_bRotLPfValid;
	protected bool m_bRotRPfValid;
	protected float m_fSlipLatL;
	protected float m_fSlipLatR;
	protected float m_fVdirL;
	protected float m_fVdirR;
	protected bool m_bSlipValid;
	protected bool m_bVdirLValid;
	protected bool m_bVdirRValid;
	protected int m_iReads;
	protected int m_iPostFrameReads;

	//------------------------------------------------------------------------------------------------
	//! Finds every candidate once. boneL / boneR: the wheel pivot bones; rotatorSuffix: appended for the kingpin pivot
	//! bones; slotL / slotR: the wheel slot names; signalName: the steering signal; wheelL / wheelR: the simulation's
	//! front wheel indices (S105: 0 = Axle0L, 1 = Axle0R, VehicleWheeledSimulation.c:23 of 1.9.0.37). A missing
	//! candidate only leaves its columns empty (header steer.channel.*); false only without a vehicle or simulation
	bool InitChannels(notnull IEntity vehicle, notnull VehicleWheeledSimulation simulation, string boneL, string boneR, string rotatorSuffix, string slotL, string slotR, string signalName, int wheelL, int wheelR)
	{
		m_Vehicle = vehicle;
		m_Simulation = simulation;
		Animation animation = vehicle.GetAnimation();
		m_bAnimation = animation != null;
		m_sSignalName = signalName;
		m_iWheelL = wheelL;
		m_iWheelR = wheelR;
		int wheels = simulation.WheelCount();
		if (m_iWheelL < 0 || m_iWheelL >= wheels)
			m_iWheelL = -1;

		if (m_iWheelR < 0 || m_iWheelR >= wheels)
			m_iWheelR = -1;

		m_aSourceNames = {};
		m_aSourceNames.Insert(boneL + rotatorSuffix);
		m_aSourceNames.Insert(boneR + rotatorSuffix);
		m_aSourceNames.Insert(boneL);
		m_aSourceNames.Insert(boneR);
		m_aSourceNames.Insert(slotL);
		m_aSourceNames.Insert(slotR);
		m_aBones = {};
		m_aEntities = {};
		m_aAxis = {};
		m_aAxisSign = {};
		m_aAxisShare = {};
		m_aValue = {};
		m_aValid = {};
		m_aReadsValid = {};

		SlotManagerComponent slots = SlotManagerComponent.Cast(vehicle.FindComponent(SlotManagerComponent));
		for (int source = 0; source < SOURCE_COUNT; source++)
		{
			int bone = -1;
			IEntity attached = null;
			string name = m_aSourceNames[source];
			if (IsBoneSource(source))
			{
				if (animation && !name.IsEmpty())
					bone = animation.GetBoneIndex(name);
			}
			else if (slots && !name.IsEmpty())
			{
				EntitySlotInfo slot = slots.GetSlotByName(name);
				if (slot)
					attached = slot.GetAttachedEntity();
			}

			m_aBones.Insert(bone);
			m_aEntities.Insert(attached);
			m_aAxis.Insert(AXIS_NOT_CHOSEN);
			m_aAxisSign.Insert(1);
			m_aAxisShare.Insert(0);
			m_aValue.Insert(0);
			m_aValid.Insert(false);
			m_aReadsValid.Insert(0);
		}

		m_Signals = SignalsManagerComponent.Cast(vehicle.FindComponent(SignalsManagerComponent));
		m_iSignal = -1;
		if (m_Signals && !signalName.IsEmpty())
			m_iSignal = m_Signals.FindSignal(signalName);

		m_sFailure = string.Empty;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Sources 0..3 are bones, 4..5 attached wheel entities
	static bool IsBoneSource(int source)
	{
		return source < SRC_ENT_L;
	}

	//------------------------------------------------------------------------------------------------
	//! Once per physics step, after it (OnProbePostSimulate): every candidate of this step
	void ReadChannels()
	{
		if (!m_Vehicle || !m_Simulation)
			return;

		m_iReads++;
		for (int source = 0; source < SOURCE_COUNT; source++)
		{
			float yaw = 0;
			bool valid = false;
			if (IsBoneSource(source))
				valid = BoneYaw(source, yaw);
			else
				valid = EntityYaw(source, yaw);

			m_aValid[source] = valid;
			m_aValue[source] = yaw;
			if (valid)
				m_aReadsValid[source] = m_aReadsValid[source] + 1;
		}

		m_bSignalValid = false;
		if (m_Signals && m_iSignal >= 0)
		{
			m_fSignal = m_Signals.GetSignalValue(m_iSignal);
			m_bSignalValid = true;
		}

		m_bSlipValid = m_iWheelL >= 0 && m_iWheelR >= 0;
		m_bVdirLValid = false;
		m_bVdirRValid = false;
		if (!m_bSlipValid)
			return;

		m_fSlipLatL = m_Simulation.WheelGetLateralSlip(m_iWheelL);
		m_fSlipLatR = m_Simulation.WheelGetLateralSlip(m_iWheelR);
		vector velL = m_Simulation.WheelGetVelocityLS(m_iWheelL);
		vector velR = m_Simulation.WheelGetVelocityLS(m_iWheelR);
		if (velL.Length() >= VDIR_MIN_SPEED)
		{
			m_fVdirL = Math.Atan2(velL[0], velL[2]);
			m_bVdirLValid = true;
		}

		if (velR.Length() >= VDIR_MIN_SPEED)
		{
			m_fVdirR = Math.Atan2(velR[0], velR[2]);
			m_bVdirRValid = true;
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Once per render frame after its physics update (OnProbePostFrame): the rotator bones only, into the *_pf columns
	void ReadPostFrame()
	{
		if (!m_Vehicle)
			return;

		m_iPostFrameReads++;
		float yaw = 0;
		m_bRotLPfValid = BoneYaw(SRC_ROT_L, yaw);
		m_fRotLPf = yaw;
		yaw = 0;
		m_bRotRPfValid = BoneYaw(SRC_ROT_R, yaw);
		m_fRotRPf = yaw;
	}

	//------------------------------------------------------------------------------------------------
	//! Yaw of a bone source from its model-space matrix (Animation.GetBoneMatrix); false when missing or unusable
	protected bool BoneYaw(int source, out float yaw)
	{
		yaw = 0;
		int bone = m_aBones[source];
		if (bone < 0)
			return false;

		Animation animation = m_Vehicle.GetAnimation();
		if (!animation)
			return false;

		vector mat[4];
		if (!animation.GetBoneMatrix(bone, mat))
			return false;

		return AxleYaw(source, mat[0], mat[1], mat[2], yaw);
	}

	//------------------------------------------------------------------------------------------------
	//! Yaw of an entity source: the attached wheel entity's world axes taken into the vehicle frame (VectorToLocal)
	protected bool EntityYaw(int source, out float yaw)
	{
		yaw = 0;
		IEntity wheel = m_aEntities[source];
		if (!wheel)
			return false;

		vector mat[4];
		wheel.GetWorldTransform(mat);
		vector axis0 = m_Vehicle.VectorToLocal(mat[0]);
		vector axis1 = m_Vehicle.VectorToLocal(mat[1]);
		vector axis2 = m_Vehicle.VectorToLocal(mat[2]);
		return AxleYaw(source, axis0, axis1, axis2, yaw);
	}

	//------------------------------------------------------------------------------------------------
	//! Picks the axle column at the first call (the axis with the largest |x| share, signed to +x; below
	//! AXLE_MIN_SHARE the source is unusable for the rest of the probe), then yaw = atan2(-a_z, a_x) of that axis
	protected bool AxleYaw(int source, vector axis0, vector axis1, vector axis2, out float yaw)
	{
		yaw = 0;
		int axis = m_aAxis[source];
		if (axis == AXIS_UNUSABLE)
			return false;

		if (axis == AXIS_NOT_CHOSEN)
		{
			float share0 = AxisShare(axis0);
			float share1 = AxisShare(axis1);
			float share2 = AxisShare(axis2);
			axis = 0;
			float share = share0;
			if (share1 > share)
			{
				axis = 1;
				share = share1;
			}

			if (share2 > share)
			{
				axis = 2;
				share = share2;
			}

			vector chosen = axis0;
			if (axis == 1)
				chosen = axis1;
			else if (axis == 2)
				chosen = axis2;

			m_aAxisShare[source] = share;
			m_aAxisSign[source] = 1;
			if (chosen[0] < 0)
				m_aAxisSign[source] = -1;

			if (share < AXLE_MIN_SHARE)
			{
				m_aAxis[source] = AXIS_UNUSABLE;
				return false;
			}

			m_aAxis[source] = axis;
		}

		vector axle = axis0;
		if (axis == 1)
			axle = axis1;
		else if (axis == 2)
			axle = axis2;

		float sign = m_aAxisSign[source];
		float ax = axle[0] * sign;
		float az = axle[2] * sign;
		if (ax * ax + az * az < 0.000001)
			return false;

		yaw = Math.Atan2(-az, ax);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! |x| / length of an axis (0 for a zero vector)
	protected static float AxisShare(vector axis)
	{
		float length = axis.Length();
		if (length <= 0)
			return 0;

		return Math.AbsFloat(axis[0]) / length;
	}

	//------------------------------------------------------------------------------------------------
	//! The step-row columns, in AddValues order
	static void AddColumns(notnull array<string> columns)
	{
		columns.Insert("wa_rot_l");
		columns.Insert("wa_rot_r");
		columns.Insert("wa_bone_l");
		columns.Insert("wa_bone_r");
		columns.Insert("wa_ent_l");
		columns.Insert("wa_ent_r");
		columns.Insert("sig_steer");
		columns.Insert("wa_rot_l_pf");
		columns.Insert("wa_rot_r_pf");
		columns.Insert("slip_lat_l");
		columns.Insert("slip_lat_r");
		columns.Insert("vdir_l");
		columns.Insert("vdir_r");
	}

	//------------------------------------------------------------------------------------------------
	//! The values of the last ReadChannels / ReadPostFrame (6 decimals; empty when not readable)
	void AddValues(notnull DC_ProbeRow row)
	{
		for (int source = 0; source < SOURCE_COUNT; source++)
		{
			AddOptional(row, m_aValid[source], m_aValue[source]);
		}

		AddOptional(row, m_bSignalValid, m_fSignal);
		AddOptional(row, m_bRotLPfValid, m_fRotLPf);
		AddOptional(row, m_bRotRPfValid, m_fRotRPf);
		AddOptional(row, m_bSlipValid, m_fSlipLatL);
		AddOptional(row, m_bSlipValid, m_fSlipLatR);
		AddOptional(row, m_bVdirLValid, m_fVdirL);
		AddOptional(row, m_bVdirRValid, m_fVdirR);
	}

	//------------------------------------------------------------------------------------------------
	protected static void AddOptional(notnull DC_ProbeRow row, bool valid, float value)
	{
		if (valid)
			row.AddFloat(value, 6);
		else
			row.AddEmpty();
	}

	//------------------------------------------------------------------------------------------------
	//! The candidates found at InitChannels and the wheel geometry (steer.* lines; appended at the probe start)
	void AppendHeaderLines(notnull array<string> lines)
	{
		lines.Insert("steer.sign_rule=rad, vehicle frame (x right, y up, z forward), positive = the wheel points right (clockwise from above), as heading_deg and a right-turn yaw rate");
		lines.Insert("steer.spin_rule=bone and entity sources: the axis of the source's frame with the largest |x| share at the first read (>= 0.9, signed to +x), yaw atan2(-a_z, a_x); the wheel spin turns the frame about that axle and leaves it unchanged");
		lines.Insert("steer.read_timing=wa_* sig_steer slip_lat_* vdir_* after each physics step (OnProbePostSimulate); wa_rot_*_pf at the last EOnPostFrame before the row");
		for (int source = 0; source < SOURCE_COUNT; source++)
		{
			string prefix = "steer.channel." + SourceKey(source);
			lines.Insert(prefix + ".name=" + m_aSourceNames[source]);
			if (IsBoneSource(source))
			{
				int bone = m_aBones[source];
				lines.Insert(prefix + ".bone_index=" + bone.ToString());
			}
			else
			{
				lines.Insert(prefix + ".entity_found=" + DC_LaunchGuard.BoolText(m_aEntities[source] != null));
			}
		}

		lines.Insert("steer.animation_found=" + DC_LaunchGuard.BoolText(m_bAnimation));
		lines.Insert("steer.signal.name=" + m_sSignalName);
		lines.Insert("steer.signal.index=" + m_iSignal.ToString());
		lines.Insert("steer.signal.manager_found=" + DC_LaunchGuard.BoolText(m_Signals != null));
		lines.Insert("steer.wheel_index_l=" + m_iWheelL.ToString());
		lines.Insert("steer.wheel_index_r=" + m_iWheelR.ToString());
		if (!m_Simulation)
			return;

		int wheels = m_Simulation.WheelCount();
		lines.Insert("steer.wheel_count=" + wheels.ToString());
		float frontZ = 0;
		float rearZ = 0;
		int rearCount = 0;
		for (int wheel = 0; wheel < wheels; wheel++)
		{
			vector position = m_Simulation.WheelGetPosition(wheel, 0);
			float radius = m_Simulation.WheelGetRadius(wheel);
			string wheelPrefix = "steer.wheel." + wheel.ToString();
			lines.Insert(wheelPrefix + ".name=" + m_Simulation.WheelGetName(wheel));
			lines.Insert(wheelPrefix + ".pos=" + DC_ProbeArea.FormatVector(position, 4));
			lines.Insert(wheelPrefix + ".radius_m=" + radius.ToString(-1, 4));
			if (wheel == m_iWheelL || wheel == m_iWheelR)
			{
				frontZ += position[2] * 0.5;
			}
			else
			{
				rearZ += position[2];
				rearCount++;
			}
		}

		if (m_iWheelL < 0 || m_iWheelR < 0 || rearCount <= 0)
			return;

		rearZ = rearZ / rearCount;
		vector posL = m_Simulation.WheelGetPosition(m_iWheelL, 0);
		vector posR = m_Simulation.WheelGetPosition(m_iWheelR, 0);
		float wheelbase = frontZ - rearZ;
		float track = Math.AbsFloat(posR[0] - posL[0]);
		lines.Insert("steer.wheelbase_m=" + wheelbase.ToString(-1, 4));
		lines.Insert("steer.track_front_m=" + track.ToString(-1, 4));
		lines.Insert("steer.wheelbase_rule=mean z of the two front wheels (wheel_index_l/r) minus the mean z of the others, WheelGetPosition(i, 0) in model space");
	}

	//------------------------------------------------------------------------------------------------
	//! The axle choice of every source and its valid reads (steer.channel.<source>.axis*; appended at the probe stop)
	void AppendSummaryLines(notnull array<string> lines)
	{
		for (int source = 0; source < SOURCE_COUNT; source++)
		{
			string prefix = "steer.channel." + SourceKey(source);
			int axis = m_aAxis[source];
			float sign = m_aAxisSign[source];
			float share = m_aAxisShare[source];
			int validReads = m_aReadsValid[source];
			lines.Insert(prefix + ".axis=" + axis.ToString());
			lines.Insert(prefix + ".axis_sign=" + sign.ToString(-1, 0));
			lines.Insert(prefix + ".axis_share=" + share.ToString(-1, 4));
			lines.Insert(prefix + ".valid_reads=" + validReads.ToString());
		}

		lines.Insert("steer.reads=" + m_iReads.ToString());
		lines.Insert("steer.postframe_reads=" + m_iPostFrameReads.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! rot_l rot_r bone_l bone_r ent_l ent_r
	static string SourceKey(int source)
	{
		if (source == SRC_ROT_L)
			return "rot_l";

		if (source == SRC_ROT_R)
			return "rot_r";

		if (source == SRC_BONE_L)
			return "bone_l";

		if (source == SRC_BONE_R)
			return "bone_r";

		if (source == SRC_ENT_L)
			return "ent_l";

		if (source == SRC_ENT_R)
			return "ent_r";

		return "unknown";
	}

	//------------------------------------------------------------------------------------------------
	//! One log line: which candidates were found
	string DescribeChannels()
	{
		string found;
		for (int source = 0; source < SOURCE_COUNT; source++)
		{
			bool present = false;
			if (IsBoneSource(source))
				present = m_aBones[source] >= 0;
			else
				present = m_aEntities[source] != null;

			if (!found.IsEmpty())
				found += " ";

			found += SourceKey(source) + ":" + DC_LaunchGuard.BoolText(present);
		}

		return string.Format("wheel-angle candidates %1, signal %2 index %3, front wheels %4/%5", found, m_sSignalName, m_iSignal, m_iWheelL, m_iWheelR);
	}
}
