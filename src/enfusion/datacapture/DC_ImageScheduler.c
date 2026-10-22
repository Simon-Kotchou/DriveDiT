// ============================================================================
// DC_ImageScheduler - the dc-2.0 image slot grid: fiducial v2 and one MakeScreenshot request in flight (DD-4-min)
// ============================================================================
//
// A DC_SessionModule (dispatch order 40) on the capture game mode. The rules are docs/capture_contract/dc-2.0.md
// section 6 "Images" (TBD-2 rate, timeout and lead from DC_Session's clocks; TBD-4 file name):
//   - FIDUCIAL: DC_Fiducial of DD-4-probe (reused unchanged) is created during readiness and shows
//     render_frame mod 2^18 first in every frame (OnFrameStart, before any row of the frame); it stays until the
//     release, so the last requests of the drain still carry it. No other widget, HUD or debug overlay is drawn;
//   - GRID: grid point g = 0, 1, 2, ... is due at the world_time_ms of physics step 0 + g x 1000 / image_hz -
//     clocks.img_lead_ms, phase-locked to the decision steps. Every grid point becomes slot g (slot_id = g, no hole:
//     an ego or camera loss ends the session instead of skipping slots). A slot is handled in the first frame whose
//     world time is at or past its due time: issued when no request is in flight (System.MakeScreenshot with the
//     extension-less path frames/f_<slot:08d>; a false return is make_failed, its issue fields kept), else
//     skipped_busy. One frame that spans several grid points issues the first and skips the others. There is no
//     fallback writer and no other file name;
//   - COMPLETION: the request in flight is resolved when one of the TBD-4 candidate files exists (as DD-4-probe's
//     DC_ImageSlotScheduler: .png, .bmp.png, .bmp, .jpg, the bare path; the suffix seen is counted in
//     session_end observed.screenshot_suffix_counts) -> heartbeat img_ok; after clocks.img_inflight_timeout_ms of
//     real time it stops blocking -> img_fail and an events row img_timeout (img_slot_id=<n>). The fiducial in the
//     file is the truth; this check is online bookkeeping only;
//   - END: OnSessionStopping stops the grid (no new slot, no skipped row); the session drains until nothing is in
//     flight or the timeout passed; a request still unresolved at the seal counts as images.pending_at_end.
// The img rows are written in slot order when each slot is handled (dc-2.0 9.4). A telemetry-only trial
// ("images": false, T0a) creates frames/ and the img stream but no fiducial and no request.
// ============================================================================

[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 image slots: world-time grid at image_hz, one MakeScreenshot request in flight, fiducial v2 (DC_Fiducial), the img stream")]
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

	[Attribute(".png,.bmp.png,.bmp,.jpg,noext", UIWidgets.EditBox, "TBD-4: file names the engine may write for an extension-less request, checked in this order (noext = the bare path; 1.8 evidence: a request ending .bmp produced *.bmp.png)", category: "Images")]
	protected string m_sScreenshotSuffixes;

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
	protected ref DC_Fiducial m_Fiducial;
	protected DC_ChunkWriter m_Img;
	protected ref DC_ProbeRow m_Row = new DC_ProbeRow();
	protected bool m_bImages;
	protected bool m_bDraining;
	protected string m_sFrameDir;
	protected float m_fPeriodMs = 100;
	protected int m_iNextSlot;					//!< next grid point = next slot_id
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
		m_fPeriodMs = 1000.0 / session.GetImageHz();

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

		m_Fiducial = new DC_Fiducial();
		if (!m_Fiducial.Create(m_iPatchPx, m_iMarginPx, m_iGapPx))
		{
			session.RefuseReadiness("fiducial: " + m_Fiducial.GetFailure());
			m_Fiducial = null;
			return false;
		}

		array<string> lines = {};
		m_Fiducial.AppendHeaderLines(lines);
		string text;
		foreach (string line : lines)
		{
			text += " " + line;
		}

		Print(LOG + "fiducial v2 created:" + text, LogLevel.NORMAL);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override bool IsSessionReady(out string waitingFor)
	{
		if (!m_bImages || (m_Fiducial && m_Fiducial.IsCreated()))
			return true;

		waitingFor = "the fiducial bands";
		return false;
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

		array<string> columns = {"slot_id", "episode_id", "ego_slot_id", "requested_world_ms", "issue_status", "issue_world_ms", "issue_render_frame", "issue_physics_step", "path", "cam_current", "cam_attached"};
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
	//! The fiducial first in every frame
	override void OnFrameStart(int renderFrame)
	{
		if (m_Fiducial)
			m_Fiducial.Show(renderFrame);
	}

	//------------------------------------------------------------------------------------------------
	//! Resolves the request in flight, then handles every grid point due by this frame's world time
	override void OnFrame(int renderFrame, float timeSlice, int worldMs)
	{
		if (!m_bImages)
			return;

		ResolveInFlight();
		if (m_bDraining || m_Session.IsStopping())
			return;

		int anchorMs = m_Session.GetStepZeroWorldMs();
		if (anchorMs < 0)
			return;

		while (DueMs(anchorMs, m_iNextSlot) <= worldMs)
		{
			if (!m_Img || m_Session.HasWriteFailure())
				return;

			HandleSlot(anchorMs, renderFrame, worldMs);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Due time of grid point g [ms, float]
	protected float DueMs(int anchorMs, int gridPoint)
	{
		return anchorMs + gridPoint * m_fPeriodMs - m_Session.GetImgLeadMs();
	}

	//------------------------------------------------------------------------------------------------
	//! One slot: issued (or make_failed) when nothing is in flight, else skipped_busy; one img row
	protected void HandleSlot(int anchorMs, int renderFrame, int worldMs)
	{
		int slot = m_iNextSlot;
		m_iNextSlot++;
		int requestedMs = Math.Round(DueMs(anchorMs, slot));
		bool camCurrent;
		bool camAttached;
		DC_RigR1 rig = m_HfRecorder.GetRig();
		if (rig)
		{
			camCurrent = rig.IsCurrent();
			camAttached = rig.IsAttached();
		}

		// One request in flight and one MakeScreenshot call per frame (so issue_render_frame strictly increases even
		// after a make_failed)
		string status = SKIPPED_BUSY;
		string relPath;
		if (!m_bInFlight && renderFrame != m_iLastAttemptFrame)
		{
			m_iLastAttemptFrame = renderFrame;
			string name = "f_" + slot.ToString(8);
			string path = m_sFrameDir + "/" + name;
			relPath = FRAMES_DIR + "/" + name;
			status = MAKE_FAILED;
			if (System.MakeScreenshot(path))
			{
				status = ISSUED;
				m_bInFlight = true;
				m_iInFlightSlot = slot;
				m_sInFlightPath = path;
				m_iInFlightTickMs = System.GetTickCount();
				m_Session.SetImagesPending(1);
			}
		}

		m_Session.CountImageSlot(status);
		m_Row.Clear();
		m_Row.AddInt(slot);
		m_Row.AddInt(m_Session.GetEpisodeId());
		m_Row.AddInt(m_Session.GetEgoSlotId());
		m_Row.AddInt(requestedMs);
		m_Row.AddString(status);
		if (status == SKIPPED_BUSY)
		{
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
		}
		else
		{
			m_Row.AddInt(worldMs);
			m_Row.AddInt(renderFrame);
			m_Row.AddInt(m_Session.GetLastPostStep());
			m_Row.AddString(relPath);
		}

		m_Row.AddBool(camCurrent);
		m_Row.AddBool(camAttached);
		if (m_Img && !m_Img.AddRow(m_Row, slot, worldMs))
			m_Session.FailWrite("img: " + m_Img.GetFailure());
	}

	//------------------------------------------------------------------------------------------------
	//! The request in flight is done when a candidate file exists, or times out after the in-flight timeout (real
	//! time): img_fail and an img_timeout event
	protected void ResolveInFlight()
	{
		if (!m_bInFlight)
			return;

		int index = FindWritten(m_sInFlightPath);
		if (index >= 0)
		{
			m_bInFlight = false;
			m_Session.SetImagesPending(0);
			m_aSuffixCounts[index] = m_aSuffixCounts[index] + 1;
			if (m_sFirstSuffix.IsEmpty())
				m_sFirstSuffix = m_aSuffixes[index];

			m_Session.CountImageResolved(true);
			return;
		}

		if (System.GetTickCount() - m_iInFlightTickMs < m_Session.GetImgInflightTimeoutMs())
			return;

		m_bInFlight = false;
		m_Session.SetImagesPending(0);
		m_Session.CountImageResolved(false);
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
		if (m_bInFlight)
			ResolveInFlight();

		return !m_bInFlight;
	}

	//------------------------------------------------------------------------------------------------
	//! images.pending_at_end and the TBD-4 outcome
	override void OnSessionClosing(string endReason)
	{
		int pending = 0;
		if (m_bInFlight)
			pending = 1;

		m_Session.SetImagesPending(pending);
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

		m_Fiducial = null;
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
