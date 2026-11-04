// ============================================================================
// DC_ImageScheduler - the dc-2.0 image slot grid: fiducial v2 and one MakeScreenshot request in flight (DD-4-min)
// ============================================================================
//
// A DC_SessionModule (dispatch order 40) on the capture game mode. The rules are docs/capture_contract/dc-2.0.md
// section 6 "Images" (TBD-2 decided: 5 Hz by default, 10 Hz only through the trial conf; timeout and lead from
// DC_Session's clocks; TBD-4 decided: the engine appends .png):
//   - FIDUCIAL: the carrier of m_eFiducialCarrier draws render_frame mod 2^18 first in every frame (OnFrameStart,
//     before any row of the frame) until the release, so the last requests of the drain still carry it. SHAPE (the
//     default, dc-2.0 freeze): DC_FiducialShape of DD-4-probe, 12 flat debug boxes m_fShapeDepthM (0.05 m) in front
//     of the rig camera, created once the rig camera exists and placed with the camera's world transform in
//     OnFrameStart, after every callback's covered steps (OnStepPost) and in EOnPostFrame (OnPostFrame), after the
//     camera applied its transform (probe run 4: 5999 of 5999 decoded, residual 0.000 px). WIDGET: DC_Fiducial (the
//     r2 3.4 ImageWidgets), a diagnostic only: widgets never reach MakeScreenshot files (run 4: 0 of 5999). No other
//     widget, HUD or debug overlay is drawn;
//   - GRID: grid point g = 0, 1, 2, ... is due at the world time of the first callback (DC_Session's grid anchor,
//     shared with the decision grid) + g x 1000 / image_hz - clocks.img_lead_ms, in integer ms. Every grid point
//     becomes slot g (slot_id = g, no hole: an ego or camera loss ends the session instead of skipping slots). A slot
//     is handled in the first frame whose world time is at or past its due time once the anchor exists: issued when
//     no request is in flight (System.MakeScreenshot with the extension-less path frames/f_<slot:08d>; a false return
//     is make_failed, its issue fields kept), else skipped_busy. One frame that spans several grid points issues the
//     first and skips the others: MakeScreenshot is called at most once per frame, so issue_render_frame strictly
//     increases, and a grid point due in the same frame after a make_failed is skipped_busy too although nothing is
//     in flight. There is no fallback writer and no other file name;
//   - COMPLETION: the request in flight is resolved in the first frame in which one of the TBD-4 candidate files
//     exists (.png, .bmp.png, .bmp, .jpg, the bare path; the suffix seen is counted in session_end
//     observed.screenshot_suffix_counts): that frame is the row's done_render_frame (in Workbench 1.8 the captured
//     frame is done_render_frame - 1) -> heartbeat img_ok; after clocks.img_inflight_timeout_ms of real time it stops
//     blocking -> img_fail and an events row img_timeout (img_slot_id=<n>). The fiducial in the file is the truth;
//     this check is online bookkeeping only;
//   - ROWS: a slot's img row is held until the slot resolves (its file seen, timed out, skipped, refused, or the
//     session sealed) and the rows are written in slot order (dc-2.0 9.4), so a skipped slot behind a request in
//     flight waits for it;
//   - END: OnSessionStopping stops the grid (no new slot, no skipped row); the session drains until nothing is in
//     flight or the timeout passed; a request still unresolved at the seal counts as images.pending_at_end and its
//     row is written with done_render_frame empty.
// A telemetry-only trial ("images": false, T0a) creates frames/ and the img stream but no fiducial and no request.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! One img row held until its slot resolves (the rows are written in slot order, dc-2.0 9.4)
class DC_ImageSlotRow
{
	int m_iSlot;
	int m_iEpisodeId;
	int m_iEgoSlotId;
	int m_iRequestedMs;
	string m_sStatus;			//!< issued | skipped_busy | make_failed
	bool m_bIssueFields;		//!< MakeScreenshot was called: issue_* and path are written
	int m_iIssueWorldMs;
	int m_iIssueFrame;
	int m_iIssueStep;
	string m_sRelPath;
	bool m_bCamCurrent;
	bool m_bCamAttached;
	bool m_bResolved;
	int m_iDoneFrame = -1;		//!< -1 = done_render_frame empty
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 image slots: world-time grid at image_hz, one MakeScreenshot request in flight, fiducial v2 (the Shape carrier), the img stream")]
class DC_ImageSchedulerClass : DC_SessionModuleClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ImageScheduler : DC_SessionModule
{
	// === FIDUCIAL (DD-4-probe DC_ProbeImages defaults; any change re-runs E-05, E-06 and E-06A) ===
	[Attribute("32", UIWidgets.Slider, "Fiducial patch size [physical px]", "8 128 1", category: "Images")]
	protected int m_iPatchPx;

	[Attribute("16", UIWidgets.Slider, "Fiducial margin from the left and right image edges [physical px]", "0 256 1", category: "Images")]
	protected int m_iMarginPx;

	[Attribute("8", UIWidgets.Slider, "Vertical gap between fiducial patches [physical px]", "0 64 1", category: "Images")]
	protected int m_iGapPx;

	[Attribute("64", UIWidgets.Slider, "Side band width of the capture class [px] (rig.band_px; the fiducial patches lie inside it)", "32 256 1", category: "Images")]
	protected int m_iBandPx;

	[Attribute(".png,.bmp.png,.bmp,.jpg,noext", UIWidgets.EditBox, "TBD-4: file names the engine may write for an extension-less request, checked in this order (noext = the bare path; decided: .png, runs 3-4; the others stay as a diagnostic)", category: "Images")]
	protected string m_sScreenshotSuffixes;

	[Attribute("1", UIWidgets.ComboBox, "Fiducial carrier (dc-2.0 freeze): SHAPE = debug Shapes in front of the rig camera, the production carrier; WIDGET = ImageWidgets, a diagnostic only (widgets never reach MakeScreenshot files, run 4); BOTH is a probe setting and refuses here", "", ParamEnumArray.FromEnum(DC_EFiducialCarrier), category: "Images")]
	protected DC_EFiducialCarrier m_eFiducialCarrier;

	[Attribute("0.05", UIWidgets.Slider, "Shape carrier depth in front of the rig camera [m] (dc-2.0 freeze: 0.05); must exceed the rig's near plane (DC_HfRecorder m_fNearPlaneM)", "0.01 100 0.01", category: "Images")]
	protected float m_fShapeDepthM;

	// === CONSTANTS ===
	static const int IMG_CHUNK_ROWS = 100;		//!< dc-2.0.streams.json img chunk_max_rows
	static const string FRAMES_DIR = "frames";
	static const string WRITER = "MakeScreenshot";
	static const string ISSUED = "issued";
	static const string SKIPPED_BUSY = "skipped_busy";
	static const string MAKE_FAILED = "make_failed";
	static const string SUFFIX_NOEXT = "noext";
	static const int DERIVATIVE_W = 512;
	static const int DERIVATIVE_H = 288;
	static const string DERIVATIVE_CODEC = "jpeg_q95";
	static const string LOG = "[DC_Images] ";

	// === STATE ===
	protected DC_Session m_Session;
	protected DC_HfRecorder m_HfRecorder;
	protected ref DC_Fiducial m_Fiducial;			//!< the WIDGET carrier (diagnostic only)
	protected ref DC_FiducialShape m_Shapes;		//!< the SHAPE carrier (production)
	protected bool m_bShapeFailed;					//!< the shape carrier refused once (readiness refused)
	protected DC_ChunkWriter m_Img;
	protected ref DC_ProbeRow m_Row = new DC_ProbeRow();
	protected bool m_bImages;
	protected bool m_bDraining;
	protected string m_sFrameDir;
	protected int m_iPeriodMs = 200;
	protected int m_iNextSlot;					//!< next grid point = next slot_id
	protected ref array<ref DC_ImageSlotRow> m_aRows = {};	//!< rows not written yet, in slot order
	protected ref array<string> m_aSuffixes = {};
	protected ref array<int> m_aSuffixCounts = {};
	protected string m_sFirstSuffix;

	// the request in flight
	protected int m_iLastAttemptFrame = -1;	//!< render frame of the last MakeScreenshot call (one per frame)
	protected bool m_bInFlight;
	protected int m_iInFlightSlot = -1;
	protected string m_sInFlightPath;
	protected int m_iInFlightTickMs;

	//------------------------------------------------------------------------------------------------
	override int GetDispatchOrder()
	{
		return 40;
	}

	//------------------------------------------------------------------------------------------------
	//! The fiducial bands (unless the trial is telemetry-only); the TBD-4 candidates
	override bool OnSessionPrepare(notnull DC_Session session)
	{
		if (SCR_Global.IsEditMode())
			return false;

		m_Session = session;
		m_HfRecorder = DC_HfRecorder.Cast(GetOwner().FindComponent(DC_HfRecorder));
		m_bImages = session.GetTrialImages();
		m_iPeriodMs = session.GetImagePeriodMs();

		array<string> suffixes = {};
		m_sScreenshotSuffixes.Split(",", suffixes, true);
		foreach (string suffix : suffixes)
		{
			m_aSuffixes.Insert(suffix.Trim());
			m_aSuffixCounts.Insert(0);
		}

		if (!m_bImages)
		{
			Print(LOG + "telemetry-only trial: no fiducial, no screenshot request", LogLevel.NORMAL);
			return true;
		}

		if (!m_HfRecorder)
		{
			session.RefuseReadiness("no DC_HfRecorder on the game mode (images need rig R1)");
			return false;
		}

		if (m_eFiducialCarrier == DC_EFiducialCarrier.BOTH)
		{
			session.RefuseReadiness("fiducial carrier BOTH is a probe setting; the capture core draws SHAPE (production) or WIDGET (diagnostic)");
			return false;
		}

		if (m_eFiducialCarrier == DC_EFiducialCarrier.SHAPE)
			return true;		// created in IsSessionReady once the rig camera exists

		Print(LOG + "WARNING: fiducial carrier WIDGET: a diagnostic only, widgets never reach MakeScreenshot files (probe run 4): the images will not decode (C13a)", LogLevel.WARNING);
		m_Fiducial = new DC_Fiducial();
		if (!m_Fiducial.Create(m_iPatchPx, m_iMarginPx, m_iGapPx))
		{
			session.RefuseReadiness("fiducial: " + m_Fiducial.GetFailure());
			m_Fiducial = null;
			return false;
		}

		array<string> lines = {};
		m_Fiducial.AppendHeaderLines(lines);
		Print(LOG + "fiducial v2 widget carrier created:" + JoinLines(lines), LogLevel.NORMAL);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override bool IsSessionReady(out string waitingFor)
	{
		waitingFor = string.Empty;
		if (!m_bImages || (m_Fiducial && m_Fiducial.IsCreated()) || (m_Shapes && m_Shapes.IsCreated()))
			return true;

		if (m_eFiducialCarrier == DC_EFiducialCarrier.SHAPE && !m_bShapeFailed && CreateShapes())
			return true;

		waitingFor = "the fiducial bands";
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! The Shape carrier on the rig camera (vFOV and near plane of the camera, the workspace size); false while the
	//! rig camera or the workspace is missing, or after a refusal (readiness refused)
	protected bool CreateShapes()
	{
		DC_RigR1 rig = m_HfRecorder.GetRig();
		if (!rig)
			return false;

		CameraBase camera = rig.GetCamera();
		WorkspaceWidget workspace = GetGame().GetWorkspace();
		if (!camera || !workspace)
			return false;

		DC_FiducialShape shapes = new DC_FiducialShape();
		if (!shapes.Create(workspace.GetWidth(), workspace.GetHeight(), camera.GetVerticalFOV(), camera.GetNearPlane(), m_iPatchPx, m_iMarginPx, m_iGapPx, m_fShapeDepthM, false))
		{
			m_bShapeFailed = true;
			m_Session.RefuseReadiness("fiducial shape carrier: " + shapes.GetFailure());
			return false;
		}

		m_Shapes = shapes;
		PlaceShapes();
		array<string> lines = {};
		m_Shapes.AppendHeaderLines(lines);
		Print(LOG + "fiducial v2 shape carrier created:" + JoinLines(lines), LogLevel.NORMAL);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The shapes follow the rig camera's world transform (the transform the camera applies, IEntity.c:203)
	protected void PlaceShapes()
	{
		if (!m_Shapes || !m_HfRecorder)
			return;

		DC_RigR1 rig = m_HfRecorder.GetRig();
		if (!rig)
			return;

		CameraBase camera = rig.GetCamera();
		if (!camera)
			return;

		vector mat[4];
		camera.GetWorldTransform(mat);
		m_Shapes.Place(mat);
	}

	//------------------------------------------------------------------------------------------------
	protected static string JoinLines(notnull array<string> lines)
	{
		string text;
		foreach (string line : lines)
		{
			text += " " + line;
		}

		return text;
	}

	//------------------------------------------------------------------------------------------------
	//! rig.writer, source_class, capture_wh, core_wh, band_px, derivative_*, fiducial, dpi_scale
	override void WriteSessionJson(string section, notnull DC_JsonWriter json)
	{
		if (section != "rig")
			return;

		int width;
		int height;
		float dpiUser = 1;
		WorkspaceWidget workspace = GetGame().GetWorkspace();
		if (workspace)
		{
			width = workspace.GetWidth();
			height = workspace.GetHeight();
			dpiUser = workspace.GetUserDPIScale();
		}

		array<int> captureWh = {width, height};
		array<int> coreWh = {width - 2 * m_iBandPx, height};
		array<int> derivativeWh = {DERIVATIVE_W, DERIVATIVE_H};
		json.AddString("writer", WRITER);
		json.AddString("source_class", GetSourceClass());
		json.AddIntArray("capture_wh", captureWh);
		json.AddIntArray("core_wh", coreWh);
		json.AddInt("band_px", m_iBandPx);
		json.AddIntArray("derivative_wh", derivativeWh);
		json.AddString("derivative_codec", DERIVATIVE_CODEC);
		json.BeginObject("fiducial");
		json.AddString("version", "v2");
		json.AddInt("bits", DC_Fiducial.CODE_BITS);
		json.AddInt("patches", DC_Fiducial.PATCH_COUNT);
		json.AddInt("bits_per_patch", DC_Fiducial.BITS_PER_PATCH);
		json.AddInt("patch_px", m_iPatchPx);
		json.AddInt("margin_px", m_iMarginPx);
		json.AddInt("gap_px", m_iGapPx);
		json.AddBool("mirrored", true);
		if (m_eFiducialCarrier == DC_EFiducialCarrier.WIDGET)
		{
			json.AddString("carrier", "widget");
			json.AddNull("depth_m");
		}
		else
		{
			json.AddString("carrier", "shape");
			json.AddFloat("depth_m", m_fShapeDepthM, 4);
		}

		json.EndObject();
		json.AddFloat("dpi_scale", dpiUser, 4);
	}

	//------------------------------------------------------------------------------------------------
	//! rig_id part: writer, source class, capture size
	override string GetRigIdPart()
	{
		int width;
		int height;
		WorkspaceWidget workspace = GetGame().GetWorkspace();
		if (workspace)
		{
			width = workspace.GetWidth();
			height = workspace.GetHeight();
		}

		return WRITER + "|" + GetSourceClass() + "|" + width.ToString() + "x" + height.ToString();
	}

	//------------------------------------------------------------------------------------------------
	//! frames/ and the img stream (owned by the session)
	override bool OnSessionOpen(notnull DC_Session session)
	{
		m_sFrameDir = session.GetSessionDir() + "/" + FRAMES_DIR;
		FileIO.MakeDirectory(m_sFrameDir);
		if (!FileIO.FileExists(m_sFrameDir))
		{
			session.FailWrite("cannot create " + m_sFrameDir);
			return false;
		}

		array<string> columns = {"slot_id", "episode_id", "ego_slot_id", "requested_world_ms", "issue_status", "issue_world_ms", "issue_render_frame", "issue_physics_step", "path", "cam_current", "cam_attached", "done_render_frame"};
		// The session keeps the only strong reference (RegisterStream); a new object assigned straight to the weak
		// member would be released at once
		DC_ChunkWriter writer = new DC_ChunkWriter();
		bool opened = writer.Open(session.GetSessionDir(), "img", columns, IMG_CHUNK_ROWS, session.GetChunkMaxMs(), session.GetKeepLogsOpen(), session.GetVerifyChunkLength());
		session.RegisterStream(writer);
		m_Img = writer;
		if (opened)
			return true;

		session.FailWrite("img: " + writer.GetFailure());
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! The fiducial first in every frame (and the shapes where the camera is now)
	override void OnFrameStart(int renderFrame)
	{
		if (m_Fiducial)
			m_Fiducial.Show(renderFrame);

		if (m_Shapes)
		{
			m_Shapes.Show(renderFrame);
			PlaceShapes();
		}
	}

	//------------------------------------------------------------------------------------------------
	//! After the callback's covered steps the ego (and the camera) moved: the shapes follow
	override void OnStepPost(int physicsStep, float timeSlice, int worldMs)
	{
		PlaceShapes();
	}

	//------------------------------------------------------------------------------------------------
	//! After the frame's physics update the camera has applied its transform (CameraBase.c:18): the last placement
	override void OnPostFrame(int renderFrame)
	{
		PlaceShapes();
	}

	//------------------------------------------------------------------------------------------------
	//! Resolves the request in flight, handles every grid point due by this frame's world time, then writes the rows
	//! that resolved, in slot order
	override void OnFrame(int renderFrame, float timeSlice, int worldMs)
	{
		if (!m_bImages)
			return;

		ResolveInFlight(renderFrame);
		int anchorMs = m_Session.GetGridAnchorWorldMs();
		if (!m_bDraining && !m_Session.IsStopping() && anchorMs >= 0)
		{
			while (DueMs(anchorMs, m_iNextSlot) <= worldMs)
			{
				if (!m_Img || m_Session.HasWriteFailure())
					return;

				HandleSlot(anchorMs, renderFrame, worldMs);
			}
		}

		FlushRows(worldMs);
	}

	//------------------------------------------------------------------------------------------------
	//! Due time of grid point g [ms]: the decision grid's anchor + g x the image period - the lead (integer ms)
	protected int DueMs(int anchorMs, int gridPoint)
	{
		return anchorMs + gridPoint * m_iPeriodMs - m_Session.GetImgLeadMs();
	}

	//------------------------------------------------------------------------------------------------
	//! One slot: issued (or make_failed) when nothing is in flight, else skipped_busy; its row is held until it resolves
	protected void HandleSlot(int anchorMs, int renderFrame, int worldMs)
	{
		int slot = m_iNextSlot;
		m_iNextSlot++;
		DC_ImageSlotRow row = new DC_ImageSlotRow();
		row.m_iSlot = slot;
		row.m_iEpisodeId = m_Session.GetEpisodeId();
		row.m_iEgoSlotId = m_Session.GetEgoSlotId();
		row.m_iRequestedMs = DueMs(anchorMs, slot);
		DC_RigR1 rig = m_HfRecorder.GetRig();
		if (rig)
		{
			row.m_bCamCurrent = rig.IsCurrent();
			row.m_bCamAttached = rig.IsAttached();
		}

		// One request in flight and one MakeScreenshot call per frame (so issue_render_frame strictly increases even
		// after a make_failed)
		row.m_sStatus = SKIPPED_BUSY;
		row.m_bResolved = true;
		if (!m_bInFlight && renderFrame != m_iLastAttemptFrame)
		{
			m_iLastAttemptFrame = renderFrame;
			string name = "f_" + slot.ToString(8);
			string path = m_sFrameDir + "/" + name;
			row.m_bIssueFields = true;
			row.m_iIssueWorldMs = worldMs;
			row.m_iIssueFrame = renderFrame;
			row.m_iIssueStep = m_Session.GetLastPostStep();
			row.m_sRelPath = FRAMES_DIR + "/" + name;
			row.m_sStatus = MAKE_FAILED;
			if (System.MakeScreenshot(path))
			{
				row.m_sStatus = ISSUED;
				row.m_bResolved = false;
				m_bInFlight = true;
				m_iInFlightSlot = slot;
				m_sInFlightPath = path;
				m_iInFlightTickMs = System.GetTickCount();
				m_Session.SetImagesPending(1);
			}
		}

		m_Session.CountImageSlot(row.m_sStatus);
		m_aRows.Insert(row);
	}

	//------------------------------------------------------------------------------------------------
	//! Writes the rows from the front of the queue while they are resolved (slot order)
	protected void FlushRows(int worldMs)
	{
		while (!m_aRows.IsEmpty() && m_aRows[0].m_bResolved)
		{
			if (!m_Img || m_Session.HasWriteFailure())
				return;

			WriteRow(m_aRows[0], worldMs);
			m_aRows.RemoveOrdered(0);
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void WriteRow(notnull DC_ImageSlotRow row, int worldMs)
	{
		m_Row.Clear();
		m_Row.AddInt(row.m_iSlot);
		m_Row.AddInt(row.m_iEpisodeId);
		m_Row.AddInt(row.m_iEgoSlotId);
		m_Row.AddInt(row.m_iRequestedMs);
		m_Row.AddString(row.m_sStatus);
		if (row.m_bIssueFields)
		{
			m_Row.AddInt(row.m_iIssueWorldMs);
			m_Row.AddInt(row.m_iIssueFrame);
			m_Row.AddInt(row.m_iIssueStep);
			m_Row.AddString(row.m_sRelPath);
		}
		else
		{
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
		}

		m_Row.AddBool(row.m_bCamCurrent);
		m_Row.AddBool(row.m_bCamAttached);
		if (row.m_iDoneFrame >= 0)
			m_Row.AddInt(row.m_iDoneFrame);
		else
			m_Row.AddEmpty();

		if (!m_Img.AddRow(m_Row, row.m_iSlot, worldMs))
			m_Session.FailWrite("img: " + m_Img.GetFailure());
	}

	//------------------------------------------------------------------------------------------------
	//! The held row of a slot, or null
	protected DC_ImageSlotRow FindRow(int slot)
	{
		foreach (DC_ImageSlotRow row : m_aRows)
		{
			if (row.m_iSlot == slot)
				return row;
		}

		return null;
	}

	//------------------------------------------------------------------------------------------------
	//! The request in flight is done in the first frame in which a candidate file exists (done_render_frame), or times
	//! out after the in-flight timeout (real time): img_fail and an img_timeout event
	protected void ResolveInFlight(int renderFrame)
	{
		if (!m_bInFlight)
			return;

		DC_ImageSlotRow row = FindRow(m_iInFlightSlot);
		int index = FindWritten(m_sInFlightPath);
		if (index >= 0)
		{
			m_bInFlight = false;
			m_Session.SetImagesPending(0);
			m_aSuffixCounts[index] = m_aSuffixCounts[index] + 1;
			if (m_sFirstSuffix.IsEmpty())
				m_sFirstSuffix = m_aSuffixes[index];

			m_Session.CountImageResolved(true);
			if (row)
			{
				row.m_iDoneFrame = renderFrame;
				row.m_bResolved = true;
			}

			return;
		}

		if (System.GetTickCount() - m_iInFlightTickMs < m_Session.GetImgInflightTimeoutMs())
			return;

		m_bInFlight = false;
		m_Session.SetImagesPending(0);
		m_Session.CountImageResolved(false);
		if (row)
			row.m_bResolved = true;

		m_Session.LogEvent("img_timeout", "img_slot_id=" + m_iInFlightSlot.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! \return index of the first TBD-4 candidate that exists for path, -1 when none does
	protected int FindWritten(string path)
	{
		for (int i = 0, count = m_aSuffixes.Count(); i < count; i++)
		{
			string candidate = path + m_aSuffixes[i];
			if (m_aSuffixes[i] == SUFFIX_NOEXT)
				candidate = path;

			if (FileIO.FileExists(candidate))
				return i;
		}

		return -1;
	}

	//------------------------------------------------------------------------------------------------
	//! No new slot from now on; the request in flight still resolves while the session drains
	override void OnSessionStopping(string endReason)
	{
		m_bDraining = true;
	}

	//------------------------------------------------------------------------------------------------
	override bool IsDrained()
	{
		return !m_bInFlight;		// OnFrame resolved it in this frame already
	}

	//------------------------------------------------------------------------------------------------
	//! images.pending_at_end and the TBD-4 outcome
	override void OnSessionClosing(string endReason)
	{
		int pending = 0;
		if (m_bInFlight)
			pending = 1;

		m_Session.SetImagesPending(pending);
		// Every held row is written now: a request still in flight keeps done_render_frame empty (pending_at_end)
		foreach (DC_ImageSlotRow held : m_aRows)
		{
			held.m_bResolved = true;
		}

		FlushRows(m_Session.GetWorldMs());
		string counts;
		for (int i = 0, count = m_aSuffixes.Count(); i < count; i++)
		{
			if (i > 0)
				counts += ";";

			int found = m_aSuffixCounts[i];
			counts += m_aSuffixes[i] + "=" + found.ToString();
		}

		m_Session.NoteObserved("screenshot_suffix_first", m_sFirstSuffix);
		m_Session.NoteObserved("screenshot_suffix_counts", counts);
		if (m_Fiducial)
		{
			int relayouts = m_Fiducial.GetRelayouts();
			m_Session.NoteObserved("fiducial_relayouts", relayouts.ToString());
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Removes the bands and restores the user DPI scale; on teardown the references only
	override void OnSessionRelease(bool teardown)
	{
		if (!teardown && m_Fiducial)
			m_Fiducial.Destroy();

		if (!teardown && m_Shapes)
			m_Shapes.Destroy();

		m_Fiducial = null;
		m_Shapes = null;
		m_aRows.Clear();
		m_Img = null;
		m_HfRecorder = null;
	}

	//------------------------------------------------------------------------------------------------
	//! rig.source_class: wb_viewport in Workbench (dims = the viewport), client_1408x720 in the client
	static string GetSourceClass()
	{
#ifdef WORKBENCH
		return "wb_viewport";
#else
		return "client_1408x720";
#endif
	}
}
