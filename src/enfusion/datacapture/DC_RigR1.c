// ============================================================================
// DC_RigR1 - rig R1: a script-created camera on the ego vehicle (DD-4-probe, r2 3.4)
// ============================================================================
//
// First version of a production class: DD-4-min reuses it, and any change re-runs E-05, E-06 and E-06A.
//
// WHAT IT DOES
//   - Attach: spawns a bare CameraBase (GetGame().SpawnEntity(CameraBase), the vanilla pattern of
//     WorkbenchGame/WorldEditor/SCR_FPSDiagnosticPlugin.c:380), attaches it to the vehicle with
//     EAddChildFlags.AUTO_TRANSFORM, sets its local transform from the mount position (vehicle-local metres, x right,
//     y up, z forward) and the mount pitch (degrees, negative = down; Math3D.AnglesToMatrix with yaw 0, roll 0), sets
//     the vertical FOV and the clip planes, remembers the current camera and makes the rig camera current
//     (CameraManager.SetCamera);
//   - Detach: makes the remembered camera current again (only while the rig camera is still current), removes the
//     child link and deletes the camera (SCR_EntityHelper.DeleteEntityAndChildren). Idempotent;
//   - IsCurrent / IsAttached / GetCameraTransform give the per-frame facts of the hf prototype (cam_current,
//     cam_attached, cam_* = CameraBase.GetWorldCameraTransform).
//
// Not done here on purpose: no HUD or overlay handling (the probe world has no player), no camera index change (the
// index is logged; r2 expects 0), no per-frame transform writes (AUTO_TRANSFORM moves the camera with the vehicle).
//
// Failure handling: Attach returns false (GetFailure) on any failure and undoes what it did. During world teardown
// (a probe stopped with reason "delete") the owner calls nothing here: the camera dies with the world.
// ============================================================================

//------------------------------------------------------------------------------------------------
class DC_RigR1
{
	static const string RIG_NAME = "R1";

	protected CameraBase m_Camera;
	protected CameraBase m_PreviousCamera;
	protected IEntity m_Vehicle;
	protected vector m_vMountPos;
	protected float m_fPitchDeg;
	protected float m_fVFovRequested;
	protected float m_fVFovRead = -1;
	protected float m_fNearM;
	protected float m_fFarM;
	protected int m_iCameraIndex = -1;
	protected bool m_bSetCameraOk;
	protected string m_sVehiclePrefab;
	protected string m_sFailure;

	//------------------------------------------------------------------------------------------------
	//! Spawns the rig camera, attaches it to vehicle at the mount and makes it current.
	//! \return false (GetFailure) on any failure; nothing is left behind then
	bool Attach(notnull IEntity vehicle, vector mountPos, float pitchDeg, float vFovDeg, float nearM, float farM)
	{
		if (m_Camera)
		{
			m_sFailure = "rig " + RIG_NAME + " is already attached";
			return false;
		}

		ArmaReforgerScripted game = GetGame();
		if (!game)
		{
			m_sFailure = "no game";
			return false;
		}

		CameraManager manager = game.GetCameraManager();
		if (!manager)
		{
			m_sFailure = "no camera manager";
			return false;
		}

		m_Vehicle = vehicle;
		m_vMountPos = mountPos;
		m_fPitchDeg = pitchDeg;
		m_fVFovRequested = vFovDeg;
		m_fNearM = nearM;
		m_fFarM = farM;
		m_sVehiclePrefab = string.Empty;
		EntityPrefabData prefabData = vehicle.GetPrefabData();
		if (prefabData)
			m_sVehiclePrefab = prefabData.GetPrefabName();

		m_Camera = CameraBase.Cast(game.SpawnEntity(CameraBase));
		if (!m_Camera)
		{
			m_sFailure = "SpawnEntity(CameraBase) returned null";
			return false;
		}

		vehicle.AddChild(m_Camera, -1, EAddChildFlags.AUTO_TRANSFORM);

		vector rotation[3];
		Math3D.AnglesToMatrix(Vector(0, pitchDeg, 0), rotation);
		vector local[4];
		local[0] = rotation[0];
		local[1] = rotation[1];
		local[2] = rotation[2];
		local[3] = mountPos;
		m_Camera.SetLocalTransform(local);

		m_Camera.SetVerticalFOV(vFovDeg);
		m_Camera.SetNearPlane(nearM);
		m_Camera.SetFarPlane(farM);

		m_PreviousCamera = manager.CurrentCamera();
		m_bSetCameraOk = manager.SetCamera(m_Camera);
		m_fVFovRead = m_Camera.GetVerticalFOV();
		m_iCameraIndex = m_Camera.GetCameraIndex();

		if (!m_bSetCameraOk || !IsCurrent())
		{
			m_sFailure = "CameraManager.SetCamera did not make the rig camera current (returned " + BoolText(m_bSetCameraOk) + ")";
			Detach();
			return false;
		}

		if (!IsAttached())
		{
			m_sFailure = "AddChild did not attach the rig camera to the vehicle";
			Detach();
			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Gives the camera back to the previous one, unlinks and deletes the rig camera. Idempotent.
	void Detach()
	{
		if (!m_Camera)
			return;

		ArmaReforgerScripted game = GetGame();
		CameraManager manager;
		if (game)
			manager = game.GetCameraManager();

		if (manager && m_PreviousCamera && manager.CurrentCamera() == m_Camera)
			manager.SetCamera(m_PreviousCamera);

		IEntity parent = m_Camera.GetParent();
		if (parent)
			parent.RemoveChild(m_Camera);

		SCR_EntityHelper.DeleteEntityAndChildren(m_Camera);
		m_Camera = null;
		m_PreviousCamera = null;
	}

	//------------------------------------------------------------------------------------------------
	//! \return true while the rig camera is the camera manager's current camera
	bool IsCurrent()
	{
		if (!m_Camera)
			return false;

		ArmaReforgerScripted game = GetGame();
		if (!game)
			return false;

		CameraManager manager = game.GetCameraManager();
		if (!manager)
			return false;

		return manager.CurrentCamera() == m_Camera;
	}

	//------------------------------------------------------------------------------------------------
	//! \return true while the rig camera's parent is the vehicle it was attached to
	bool IsAttached()
	{
		if (!m_Camera || !m_Vehicle)
			return false;

		return m_Camera.GetParent() == m_Vehicle;
	}

	//------------------------------------------------------------------------------------------------
	CameraBase GetCamera()
	{
		return m_Camera;
	}

	//------------------------------------------------------------------------------------------------
	//! Camera world transform (CameraBase.GetWorldCameraTransform); all zero without a camera
	void GetCameraTransform(out vector mat[4])
	{
		if (!m_Camera)
		{
			mat[0] = vector.Zero;
			mat[1] = vector.Zero;
			mat[2] = vector.Zero;
			mat[3] = vector.Zero;
			return;
		}

		m_Camera.GetWorldCameraTransform(mat);
	}

	//------------------------------------------------------------------------------------------------
	string GetFailure()
	{
		return m_sFailure;
	}

	//------------------------------------------------------------------------------------------------
	//! \return vertical FOV read back from the camera after Attach [deg], -1 before
	float GetVerticalFovRead()
	{
		return m_fVFovRead;
	}

	//------------------------------------------------------------------------------------------------
	//! Appends the rig.* key=value lines to lines
	void AppendHeaderLines(notnull array<string> lines)
	{
		int renderW;
		int renderH;
		System.GetRenderingResolution(renderW, renderH);
		int nativeW;
		int nativeH;
		System.GetNativeResolution(nativeW, nativeH);

		lines.Insert("rig.name=" + RIG_NAME);
		lines.Insert("rig.mount_pos=" + FormatVector(m_vMountPos, 3));
		lines.Insert("rig.pitch_deg=" + m_fPitchDeg.ToString(-1, 3));
		lines.Insert("rig.vfov_requested=" + m_fVFovRequested.ToString(-1, 3));
		lines.Insert("rig.vfov_read=" + m_fVFovRead.ToString(-1, 6));
		lines.Insert("rig.near_m=" + m_fNearM.ToString(-1, 3));
		lines.Insert("rig.far_m=" + m_fFarM.ToString(-1, 3));
		lines.Insert("rig.camera_index=" + m_iCameraIndex.ToString());
		lines.Insert("rig.set_camera_ok=" + BoolText(m_bSetCameraOk));
		lines.Insert("rig.vehicle_prefab=" + m_sVehiclePrefab);
		lines.Insert("rig.rendering_w=" + renderW.ToString());
		lines.Insert("rig.rendering_h=" + renderH.ToString());
		lines.Insert("rig.native_w=" + nativeW.ToString());
		lines.Insert("rig.native_h=" + nativeH.ToString());

		// Information only: the pinhole fy that the logged vFOV gives at the rendering height
		if (renderH > 0 && m_fVFovRead > 0)
		{
			float fy = 0.5 * renderH / Math.Tan(0.5 * m_fVFovRead * Math.DEG2RAD);
			lines.Insert("rig.fy_px_at_rendering_h=" + fy.ToString(-1, 3));
		}
	}

	//------------------------------------------------------------------------------------------------
	//! \return "x y z" with the given number of decimals (header and parameter values; never a CSV field)
	static string FormatVector(vector value, int decimals)
	{
		float x = value[0];
		float y = value[1];
		float z = value[2];
		return x.ToString(-1, decimals) + " " + y.ToString(-1, decimals) + " " + z.ToString(-1, decimals);
	}

	//------------------------------------------------------------------------------------------------
	protected static string BoolText(bool value)
	{
		if (value)
			return "1";

		return "0";
	}
}
