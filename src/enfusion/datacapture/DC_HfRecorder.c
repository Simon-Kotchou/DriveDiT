// ============================================================================
// DC_HfRecorder - rig R1 in the dc-2.0 session and the hf stream (DD-4-min)
// ============================================================================
//
// A DC_SessionModule (dispatch order 30) on the capture game mode.
//
// RIG R1 (DC_RigR1 of DD-4-probe, reused unchanged, so its E-05 / E-06 / E-06A qualification holds): during
//   readiness, once DC_EgoSession has spawned the ego, the rig camera is attached to it (mount, pitch, vFOV and clip
//   planes are this component's attributes, with the DD-4-probe defaults) and made current. An attach failure refuses
//   readiness (camera init failure fails the session before any directory exists); readiness waits until the
//   camera is current and attached. While capturing, a frame whose rig camera is not current or not attached ends
//   the session with camera_lost (DC_ImageScheduler logs cam_current / cam_attached per slot).
//
// HF STREAM (dc-2.0 9.3, every render frame from the open to the seal, the drain included, so hf.render_frame of
//   the last row is session_end end.render_frame): render_frame = BaseWorld.GetFrameNumber(), world_time_ms,
//   last_physics_step (the most recent OnStepPost, -1 before the first), paused (ChimeraWorld.IsGameTimePaused),
//   the camera's world transform (CameraBase.GetWorldCameraTransform), the ego's world transform, fps
//   (System.GetFPS()). A frame number that does not advance is not written again (counted in session_end
//   observed.hf_frame_repeats).
//
// session.json rig.*: rig_name, mount, vfov_deg (read back from the camera); rig_id part: rig, vehicle, mount,
// pitch, vFOV. Release: the previous camera is current again and the rig camera is deleted; on world teardown the
// camera dies with the world and only the reference is dropped.
// ============================================================================

[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 rig R1 camera on the ego (DC_RigR1) and the hf stream, one row per render frame")]
class DC_HfRecorderClass : DC_SessionModuleClass
{
}

//------------------------------------------------------------------------------------------------
class DC_HfRecorder : DC_SessionModule
{
	// === RIG (DD-4-probe DC_ProbeImages defaults; any change re-runs E-05, E-06 and E-06A) ===
	[Attribute("0 1.25 0.3", UIWidgets.Coords, "Camera mount in the ego's local frame [m]: x right, y up, z forward [unverified: tune from the E-04 bounds and the first frames]", category: "Rig")]
	protected vector m_vMountPos;

	[Attribute("-4", UIWidgets.Slider, "Camera pitch [deg], negative = down (r2 3.4: -3 to -5)", "-30 30 0.5", category: "Rig")]
	protected float m_fMountPitchDeg;

	[Attribute("50", UIWidgets.Slider, "Vertical field of view [deg] (r2 3.4)", "10 120 1", category: "Rig")]
	protected float m_fVerticalFovDeg;

	[Attribute("0.1", UIWidgets.Slider, "Near clip plane [m]", "0.01 10 0.01", category: "Rig")]
	protected float m_fNearPlaneM;

	[Attribute("3000", UIWidgets.Slider, "Far clip plane [m]", "100 20000 10", category: "Rig")]
	protected float m_fFarPlaneM;

	// === CONSTANTS ===
	static const int HF_CHUNK_ROWS = 600;		//!< dc-2.0.streams.json hf chunk_max_rows
	static const string LOG = "[DC_Hf] ";

	// === STATE ===
	protected DC_Session m_Session;
	protected ref DC_RigR1 m_Rig;
	protected DC_ChunkWriter m_Hf;
	protected ref DC_ProbeRow m_Row = new DC_ProbeRow();
	protected int m_iLastFrame = -1;
	protected int m_iFrameRepeats;
	protected string m_sVehicleName;

	//------------------------------------------------------------------------------------------------
	override int GetDispatchOrder()
	{
		return 30;
	}

	//------------------------------------------------------------------------------------------------
	override bool OnSessionPrepare(notnull DC_Session session)
	{
		if (SCR_Global.IsEditMode())
			return false;

		m_Session = session;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Attaches rig R1 as soon as the ego exists; ready while the camera is current and attached
	override bool IsSessionReady(out string waitingFor)
	{
		IEntity ego = m_Session.GetEgo();
		if (!ego)
		{
			waitingFor = "the ego";
			return false;
		}

		if (!m_Rig)
		{
			m_Rig = new DC_RigR1();
			if (!m_Rig.Attach(ego, m_vMountPos, m_fMountPitchDeg, m_fVerticalFovDeg, m_fNearPlaneM, m_fFarPlaneM))
			{
				m_Session.RefuseReadiness("rig " + DC_RigR1.RIG_NAME + ": " + m_Rig.GetFailure());
				m_Rig = null;
				waitingFor = "the rig camera";
				return false;
			}

			string prefab = m_Session.GetEgoPrefab();
			m_sVehicleName = FilePath.StripExtension(FilePath.StripPath(prefab));
			array<string> lines = {};
			m_Rig.AppendHeaderLines(lines);
			Print(LOG + "rig " + DC_RigR1.RIG_NAME + " attached to " + m_sVehicleName + ": " + JoinLines(lines), LogLevel.NORMAL);
		}

		if (!m_Rig.IsCurrent() || !m_Rig.IsAttached())
		{
			waitingFor = string.Format("the rig camera (current %1, attached %2)", DC_LaunchGuard.BoolText(m_Rig.IsCurrent()), DC_LaunchGuard.BoolText(m_Rig.IsAttached()));
			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! rig.rig_name, rig.mount, rig.vfov_deg
	override void WriteSessionJson(string section, notnull DC_JsonWriter json)
	{
		if (section != "rig")
			return;

		json.AddString("rig_name", DC_RigR1.RIG_NAME + "-" + m_sVehicleName);
		json.BeginObject("mount");
		json.AddFloat("fwd_m", m_vMountPos[2], 4);
		json.AddFloat("up_m", m_vMountPos[1], 4);
		json.AddFloat("right_m", m_vMountPos[0], 4);
		json.AddFloat("pitch_deg", m_fMountPitchDeg, 4);
		json.EndObject();
		float vFovRead = m_fVerticalFovDeg;
		if (m_Rig)
			vFovRead = m_Rig.GetVerticalFovRead();

		json.AddFloat("vfov_deg", vFovRead, 4);
	}

	//------------------------------------------------------------------------------------------------
	//! rig_id part: rig, vehicle, mount x,y,z, pitch, vFOV read back
	override string GetRigIdPart()
	{
		float vFovRead = m_fVerticalFovDeg;
		if (m_Rig)
			vFovRead = m_Rig.GetVerticalFovRead();

		string mount = DC_RigR1.FormatVector(m_vMountPos, 3);
		mount.Replace(" ", ",");
		return DC_RigR1.RIG_NAME + "|" + m_sVehicleName + "|m" + mount + "|p" + m_fMountPitchDeg.ToString(-1, 2) + "|v" + vFovRead.ToString(-1, 2);
	}

	//------------------------------------------------------------------------------------------------
	//! The hf stream (owned by the session)
	override bool OnSessionOpen(notnull DC_Session session)
	{
		array<string> columns = {"render_frame", "world_time_ms", "last_physics_step", "paused"};
		DC_ProbeCsv.AddMat34Columns(columns, "cam");
		DC_ProbeCsv.AddMat34Columns(columns, "veh");
		columns.Insert("fps");

		m_Hf = new DC_ChunkWriter();
		bool opened = m_Hf.Open(session.GetSessionDir(), "hf", columns, HF_CHUNK_ROWS, session.GetChunkMaxMs(), session.GetKeepLogsOpen(), session.GetVerifyChunkLength());
		session.RegisterStream(m_Hf);
		if (opened)
			return true;

		session.FailWrite("hf: " + m_Hf.GetFailure());
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Camera check, then one hf row
	override void OnFrame(int renderFrame, float timeSlice, int worldMs)
	{
		if (!m_Rig || !m_Rig.IsCurrent() || !m_Rig.IsAttached())
		{
			bool current;
			bool attached;
			if (m_Rig)
			{
				current = m_Rig.IsCurrent();
				attached = m_Rig.IsAttached();
			}

			m_Session.Fail(DC_Session.END_CAMERA_LOST, string.Format("rig camera current %1, attached %2 at render frame %3", DC_LaunchGuard.BoolText(current), DC_LaunchGuard.BoolText(attached), renderFrame));
			return;
		}

		if (renderFrame <= m_iLastFrame)
		{
			m_iFrameRepeats++;
			return;
		}

		m_iLastFrame = renderFrame;
		vector camera[4];
		m_Rig.GetCameraTransform(camera);
		vector vehicle[4];
		IEntity ego = m_Session.GetEgo();
		if (ego)
			ego.GetWorldTransform(vehicle);

		m_Row.Clear();
		m_Row.AddInt(renderFrame);
		m_Row.AddInt(worldMs);
		m_Row.AddInt(m_Session.GetLastPostStep());
		m_Row.AddBool(m_Session.IsPaused());
		m_Row.AddMat34(camera, 6, 4);
		m_Row.AddMat34(vehicle, 6, 4);
		m_Row.AddFloat(System.GetFPS(), 3);
		if (m_Hf && !m_Hf.AddRow(m_Row, renderFrame, worldMs))
			m_Session.FailWrite("hf: " + m_Hf.GetFailure());
	}

	//------------------------------------------------------------------------------------------------
	override void OnSessionClosing(string endReason)
	{
		m_Session.NoteObserved("hf_frame_repeats", m_iFrameRepeats.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! The previous camera current again, the rig camera deleted; on teardown the reference only
	override void OnSessionRelease(bool teardown)
	{
		if (!teardown && m_Rig)
			m_Rig.Detach();

		m_Rig = null;
		m_Hf = null;
	}

	//------------------------------------------------------------------------------------------------
	//! The rig, null before the attach and after the release (DC_ImageScheduler reads cam_current / cam_attached)
	DC_RigR1 GetRig()
	{
		return m_Rig;
	}

	//------------------------------------------------------------------------------------------------
	protected static string JoinLines(notnull array<string> lines)
	{
		string text;
		foreach (string line : lines)
		{
			if (!text.IsEmpty())
				text += " ";

			text += line;
		}

		return text;
	}
}
