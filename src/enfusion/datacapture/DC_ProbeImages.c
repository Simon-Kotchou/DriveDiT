// ============================================================================
// DC_ProbeImages - T0-ENG probe E-05 / E-06: rig R1, fiducial v2 and 10 Hz screenshots (DD-4-probe)
// ============================================================================
//
// Probe id "e05" (data for E-05 and E-06; the analysis is tools/probes/t0eng/e05_images.py and e06_fiducial.py).
// Exclusive, needs DC_Ego and a physics-step source. Runs on the probe game mode under DC_ProbeRunner, which owns the
// launch guard, the play directory, run_header.txt and run_footer.txt.
//
// WHAT IT DOES
//   - OnProbeStart: attaches rig R1 to DC_Ego (DC_RigR1: script camera, AUTO_TRANSFORM, current, vFOV), creates the
//     fiducial bands (DC_Fiducial, with m_bFiducial), creates <play>/frames/, opens e05_requests.csv and
//     e05_frames.csv, starts the image slot grid (DC_ImageSlotScheduler) at the current world time, and appends the
//     rig.*, fiducial.* and area.* lines to the run header (they are only known once the rig and the bands exist);
//   - every render frame (OnProbeFrame), in this order: DC_Fiducial.Show(render frame); the scheduler (completion
//     poll, timeouts, the due request: one System.MakeScreenshot in flight, extension-less path
//     frames/f_<slot:08d>); one e05_frames.csv row (the hf prototype: camera and vehicle transforms, fps);
//   - with m_bDrive, a shuttle along the test area on the physics steps: reset (DC_ProbeActuator.Teleport to the
//     start point 2 x m_fAreaMarginM along the centre line, as DC_ProbeSequences, so the start lies inside the
//     margin-inset area), settle (brake and handbrake for m_fSettleS), drive (speed hold m_fDriveKmh, steering 0)
//     until the far end or the area edge, stop (brake to < 1 km/h), and again. Without m_bDrive the car is not
//     commanded and the segment is "static". The teleport itself runs in OnProbeFrame, never inside a physics step
//     (the rule of DC_ProbeSequences): the reset segment waits for it;
//   - after m_fDurationS of world time no new slot is issued; the requests still in flight are drained with the
//     camera and the bands kept (DC_ImageSlotScheduler.StopIssuing), then the probe is done. OnProbeStop resolves
//     what is left (only on a timeout or a Stop), parks the car, removes the bands and gives the camera back.
//
// The E-05 criteria (missing <= 1%, "Screenshot saved" count = files, FPS p5 >= 30, latency, MB/frame, engine-written
// name) and E-06 (decode >= 99.5% over >= 600 frames, offset) are evaluated offline from these files.
//
// Failure handling: OnProbeStart returns false (footer status failed) when DC_Ego has no CarControllerComponent while
// m_bDrive is set, when the shuttle start point is not inside the margin-inset area, when the rig cannot attach, the
// bands cannot be created, frames/ cannot be created or a CSV cannot be opened; it undoes what it did. A failed CSV
// write stops the probe through the runner (reason write_fail).
// Reason "delete" (world teardown) touches no camera, widget, setting or entity: it only drops references.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! One image slot on its way to its e05_requests / e06a_requests row: in flight, or resolved at once.
class DC_ImageSlotRequest
{
	int m_iSlot;
	float m_fRequestedMs;
	string m_sIssueStatus;		//!< issued | skipped_busy | skipped_late | make_failed
	bool m_bAttempted;			//!< MakeScreenshot was called (issued or make_failed): the issue_* fields are set
	float m_fIssueMs;
	int m_iIssueFrame;
	int m_iIssuePhysicsStep;
	int m_iIssueTickMs;
	string m_sPath;				//!< full request path without extension
	string m_sRelPath;			//!< request path relative to the play dir, without extension
	bool m_bMakeOk;
	bool m_bCamCurrent;
	bool m_bCamAttached;
	int m_iInFlightAtIssue;
	bool m_bBacklog;
	ref array<string> m_aExtra = {};
}

//------------------------------------------------------------------------------------------------
//! World-time slot grid with one request in flight (or unlimited, for backlog phases), shared by e05 and e06a.
//! A slot is due at startWorldMs + n * 1000 / imageHz; the request is issued at the first frame whose world time is
//! at or past the slot (r2 3.5). One CSV row per slot, written when the slot resolves (so rows can be out of slot
//! order: a skipped slot resolves at once while an earlier one is still in flight).
class DC_ImageSlotScheduler
{
	static const string ISSUED = "issued";
	static const string SKIPPED_BUSY = "skipped_busy";		//!< the previous request was still in flight at the slot
	static const string SKIPPED_LATE = "skipped_late";		//!< one frame spanned several slots: all but the last are skipped
	static const string MAKE_FAILED = "make_failed";		//!< System.MakeScreenshot returned false
	static const string DONE = "done";
	static const string TIMEOUT = "timeout";
	static const string NONE = "none";
	//! found_suffix of a file written exactly at the extension-less request path
	static const string SUFFIX_NOEXT = "noext";
	//! In-flight requests polled per frame, oldest first (each poll is up to 5 FileExists; backlog keeps many in flight)
	static const int POLL_PER_FRAME = 8;

	//! Names the engine may write for an extension-less request, checked in this order (TBD-4; 1.8 evidence: a
	//! request ending .bmp produced <request>.png, i.e. *.bmp.png)
	protected ref array<string> m_aSuffixes = {".png", ".bmp.png", ".bmp", ".jpg", ""};

	protected string m_sFrameDir;
	protected string m_sRelPrefix;
	protected float m_fPeriodMs = 100;
	protected float m_fStartMs;
	protected int m_iGridIndex;					//!< next grid point, due at m_fStartMs + m_iGridIndex * m_fPeriodMs
	protected int m_iNextSlot;
	protected int m_iTimeoutMs = 2000;
	protected bool m_bBacklog;
	protected bool m_bReanchor;					//!< leaving backlog: realign the grid at the next frame
	protected bool m_bInitialized;
	protected bool m_bFinished;
	protected bool m_bDraining;					//!< StopIssuing was called: no new slot, in-flight requests still resolve
	protected DC_ProbeCsv m_Requests;			//!< owned by the probe context
	protected ref DC_ProbeRow m_Row = new DC_ProbeRow();
	protected ref array<ref DC_ImageSlotRequest> m_aInFlight = {};	//!< oldest first
	protected ref array<string> m_aExtra = {};

	protected int m_iLastFrame;
	protected float m_fLastWorldMs;
	protected int m_iLastTickMs;

	protected int m_iIssued;
	protected int m_iDone;
	protected int m_iSkippedBusy;
	protected int m_iSkippedLate;
	protected int m_iTimedOut;
	protected int m_iMakeFailed;

	//------------------------------------------------------------------------------------------------
	//! frameDir = <play dir>/frames; requests are frameDir + "/f_" + slot.ToString(8), no extension.
	//! requests must have been opened with AddRequestColumns (plus the names of SetExtraValues, if any).
	void Init(string frameDir, float imageHz, float startWorldMs, int inFlightTimeoutMs, notnull DC_ProbeCsv requests)
	{
		m_sFrameDir = frameDir;
		m_sRelPrefix = FilePath.StripPath(frameDir);
		if (imageHz > 0)
			m_fPeriodMs = 1000.0 / imageHz;

		m_fStartMs = startWorldMs;
		m_iTimeoutMs = inFlightTimeoutMs;
		m_Requests = requests;
		m_iGridIndex = 0;
		m_iNextSlot = 0;
		m_bInitialized = true;
		m_bFinished = false;
		m_bDraining = false;
	}

	//------------------------------------------------------------------------------------------------
	//! No new slot from now on (no skipped rows either). OnFrame keeps polling and expiring the requests in flight
	//! until IsDrained; the probe keeps the camera and the bands until then, so the last files are still written by
	//! the rig camera before the probe ends. Idempotent.
	//! \return true on the first call
	bool StopIssuing()
	{
		if (m_bDraining)
			return false;

		m_bDraining = true;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! true once StopIssuing was called and no request is in flight any more (each resolved as done or timeout)
	bool IsDrained()
	{
		return m_bDraining && m_aInFlight.IsEmpty();
	}

	//------------------------------------------------------------------------------------------------
	//! true: a request every frame, no in-flight limit (slot = the frame). false: back to the grid, realigned to the
	//! first grid point at or after the next frame's world time.
	void SetBacklog(bool unlimitedInFlight)
	{
		if (m_bBacklog && !unlimitedInFlight)
			m_bReanchor = true;

		m_bBacklog = unlimitedInFlight;
	}

	//------------------------------------------------------------------------------------------------
	//! Values appended to every request row written from now on (copied at issue time; e06a: fps_setting, segment)
	void SetExtraValues(notnull array<string> values)
	{
		m_aExtra.Clear();
		foreach (string value : values)
		{
			m_aExtra.Insert(value);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Once per EOnFrame after DC_Fiducial.Show: polls completions, times out, then issues the due slot or records
	//! the skipped slots.
	void OnFrame(float worldMs, int renderFrame, int physicsStep, int tickMs, bool camCurrent, bool camAttached)
	{
		if (!m_bInitialized || m_bFinished)
			return;

		m_iLastFrame = renderFrame;
		m_fLastWorldMs = worldMs;
		m_iLastTickMs = tickMs;

		Poll(POLL_PER_FRAME, renderFrame, worldMs, tickMs);
		Expire(renderFrame, worldMs, tickMs);

		if (m_bDraining)
			return;

		if (m_bBacklog)
		{
			Issue(worldMs, worldMs, renderFrame, physicsStep, tickMs, camCurrent, camAttached);
			return;
		}

		if (m_bReanchor)
		{
			m_bReanchor = false;
			int index = Math.Floor((worldMs - m_fStartMs) / m_fPeriodMs);
			if (DueMs(index) < worldMs)
				index++;

			if (index > m_iGridIndex)
				m_iGridIndex = index;
		}

		if (DueMs(m_iGridIndex) > worldMs)
			return;

		// One frame spanned several grid points: every one but the last is skipped.
		while (DueMs(m_iGridIndex + 1) <= worldMs)
		{
			DC_ImageSlotRequest late = NewRequest(DueMs(m_iGridIndex), SKIPPED_LATE, camCurrent, camAttached);
			m_iSkippedLate++;
			WriteRequestRow(late, NONE, false, 0, 0, 0, string.Empty);
			m_iGridIndex++;
		}

		float requestedMs = DueMs(m_iGridIndex);
		m_iGridIndex++;

		if (!m_aInFlight.IsEmpty())
		{
			DC_ImageSlotRequest busy = NewRequest(requestedMs, SKIPPED_BUSY, camCurrent, camAttached);
			m_iSkippedBusy++;
			WriteRequestRow(busy, NONE, false, 0, 0, 0, string.Empty);
			return;
		}

		Issue(requestedMs, worldMs, renderFrame, physicsStep, tickMs, camCurrent, camAttached);
	}

	//------------------------------------------------------------------------------------------------
	//! \return world time of grid point index [ms]
	protected float DueMs(int index)
	{
		return m_fStartMs + index * m_fPeriodMs;
	}

	//------------------------------------------------------------------------------------------------
	protected DC_ImageSlotRequest NewRequest(float requestedMs, string issueStatus, bool camCurrent, bool camAttached)
	{
		DC_ImageSlotRequest request = new DC_ImageSlotRequest();
		request.m_iSlot = m_iNextSlot;
		m_iNextSlot++;
		request.m_fRequestedMs = requestedMs;
		request.m_sIssueStatus = issueStatus;
		request.m_bCamCurrent = camCurrent;
		request.m_bCamAttached = camAttached;
		request.m_iInFlightAtIssue = m_aInFlight.Count();
		request.m_bBacklog = m_bBacklog;
		foreach (string value : m_aExtra)
		{
			request.m_aExtra.Insert(value);
		}

		return request;
	}

	//------------------------------------------------------------------------------------------------
	protected void Issue(float requestedMs, float worldMs, int renderFrame, int physicsStep, int tickMs, bool camCurrent, bool camAttached)
	{
		DC_ImageSlotRequest request = NewRequest(requestedMs, ISSUED, camCurrent, camAttached);
		string name = "f_" + request.m_iSlot.ToString(8);
		request.m_bAttempted = true;
		request.m_fIssueMs = worldMs;
		request.m_iIssueFrame = renderFrame;
		request.m_iIssuePhysicsStep = physicsStep;
		request.m_iIssueTickMs = tickMs;
		request.m_sPath = m_sFrameDir + "/" + name;
		request.m_sRelPath = m_sRelPrefix + "/" + name;
		request.m_bMakeOk = System.MakeScreenshot(request.m_sPath);

		if (!request.m_bMakeOk)
		{
			request.m_sIssueStatus = MAKE_FAILED;
			m_iMakeFailed++;
			WriteRequestRow(request, NONE, false, 0, 0, 0, string.Empty);
			return;
		}

		m_iIssued++;
		m_aInFlight.Insert(request);
	}

	//------------------------------------------------------------------------------------------------
	//! Resolves in-flight requests whose file exists, oldest first; at most limit polls (limit < 0: all)
	protected void Poll(int limit, int renderFrame, float worldMs, int tickMs)
	{
		int polled = 0;
		int i = 0;
		while (i < m_aInFlight.Count())
		{
			if (limit >= 0 && polled >= limit)
				return;

			polled++;
			DC_ImageSlotRequest request = m_aInFlight[i];
			string suffix = FindWritten(request.m_sPath);
			if (suffix.IsEmpty())
			{
				i++;
				continue;
			}

			m_aInFlight.RemoveOrdered(i);
			m_iDone++;
			WriteRequestRow(request, DONE, true, renderFrame, worldMs, tickMs, suffix);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Resolves requests older than the in-flight timeout (real time): done if the file is there now, else timeout
	protected void Expire(int renderFrame, float worldMs, int tickMs)
	{
		int i = 0;
		while (i < m_aInFlight.Count())
		{
			DC_ImageSlotRequest request = m_aInFlight[i];
			if (tickMs - request.m_iIssueTickMs < m_iTimeoutMs)
			{
				i++;
				continue;
			}

			m_aInFlight.RemoveOrdered(i);
			string suffix = FindWritten(request.m_sPath);
			if (!suffix.IsEmpty())
			{
				m_iDone++;
				WriteRequestRow(request, DONE, true, renderFrame, worldMs, tickMs, suffix);
				continue;
			}

			m_iTimedOut++;
			WriteRequestRow(request, TIMEOUT, false, 0, 0, 0, string.Empty);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! \return the suffix of the first engine-written candidate that exists (SUFFIX_NOEXT for the bare path), or empty
	protected string FindWritten(string path)
	{
		foreach (string suffix : m_aSuffixes)
		{
			if (!FileIO.FileExists(path + suffix))
				continue;

			if (suffix.IsEmpty())
				return SUFFIX_NOEXT;

			return suffix;
		}

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	protected void WriteRequestRow(notnull DC_ImageSlotRequest request, string doneStatus, bool hasDone, int doneFrame, float doneMs, int doneTickMs, string suffix)
	{
		if (!m_Requests)
			return;

		m_Row.Clear();
		m_Row.AddInt(request.m_iSlot);
		m_Row.AddFloat(request.m_fRequestedMs, 3);
		m_Row.AddString(request.m_sIssueStatus);
		if (request.m_bAttempted)
		{
			m_Row.AddFloat(request.m_fIssueMs, 3);
			m_Row.AddInt(request.m_iIssueFrame);
			m_Row.AddInt(request.m_iIssuePhysicsStep);
			m_Row.AddInt(request.m_iIssueTickMs);
			m_Row.AddString(request.m_sRelPath);
		}
		else
		{
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
		}

		m_Row.AddBool(request.m_bMakeOk);
		m_Row.AddBool(request.m_bCamCurrent);
		m_Row.AddBool(request.m_bCamAttached);
		m_Row.AddInt(request.m_iInFlightAtIssue);
		m_Row.AddBool(request.m_bBacklog);
		m_Row.AddString(doneStatus);
		if (hasDone)
		{
			m_Row.AddInt(doneFrame);
			m_Row.AddFloat(doneMs, 3);
			m_Row.AddInt(doneTickMs);
			m_Row.AddString(suffix);
		}
		else
		{
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
			m_Row.AddEmpty();
		}

		foreach (string value : request.m_aExtra)
		{
			m_Row.AddString(value);
		}

		m_Requests.WriteRow(m_Row);
	}

	//------------------------------------------------------------------------------------------------
	int GetInFlight()
	{
		return m_aInFlight.Count();
	}

	//------------------------------------------------------------------------------------------------
	//! Resolves what is still in flight (one last poll, then timeout rows). Idempotent; OnFrame does nothing after it.
	void Finish()
	{
		if (!m_bInitialized || m_bFinished)
			return;

		Poll(-1, m_iLastFrame, m_fLastWorldMs, m_iLastTickMs);
		foreach (DC_ImageSlotRequest request : m_aInFlight)
		{
			m_iTimedOut++;
			WriteRequestRow(request, TIMEOUT, false, 0, 0, 0, string.Empty);
		}

		m_aInFlight.Clear();
		m_bFinished = true;
	}

	//------------------------------------------------------------------------------------------------
	//! Drops every reference without writing (world teardown)
	void Abandon()
	{
		m_aInFlight.Clear();
		m_Requests = null;
		m_bFinished = true;
	}

	//------------------------------------------------------------------------------------------------
	int GetIssued()
	{
		return m_iIssued;
	}

	//------------------------------------------------------------------------------------------------
	int GetDone()
	{
		return m_iDone;
	}

	//------------------------------------------------------------------------------------------------
	//! \return skipped slots, busy and late
	int GetSkipped()
	{
		return m_iSkippedBusy + m_iSkippedLate;
	}

	//------------------------------------------------------------------------------------------------
	int GetSkippedLate()
	{
		return m_iSkippedLate;
	}

	//------------------------------------------------------------------------------------------------
	int GetTimedOut()
	{
		return m_iTimedOut;
	}

	//------------------------------------------------------------------------------------------------
	int GetMakeFailed()
	{
		return m_iMakeFailed;
	}

	//------------------------------------------------------------------------------------------------
	//! The request columns (aligned with the dc-2.0 img stream); a probe appends its extra names after them
	static void AddRequestColumns(notnull array<string> columns)
	{
		columns.Insert("slot_id");
		columns.Insert("requested_world_ms");
		columns.Insert("issue_status");
		columns.Insert("issue_world_ms");
		columns.Insert("issue_render_frame");
		columns.Insert("issue_physics_step");
		columns.Insert("issue_tick_ms");
		columns.Insert("path");
		columns.Insert("make_ok");
		columns.Insert("cam_current");
		columns.Insert("cam_attached");
		columns.Insert("inflight_at_issue");
		columns.Insert("backlog");
		columns.Insert("done_status");
		columns.Insert("done_render_frame");
		columns.Insert("done_world_ms");
		columns.Insert("done_tick_ms");
		columns.Insert("found_suffix");
	}
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "T0-ENG probe E-05/E-06: rig R1 camera on DC_Ego, fiducial v2 and 10 Hz MakeScreenshot requests with per-request and per-frame CSVs")]
class DC_ProbeImagesClass : DC_ProbeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeImages : DC_ProbeComponent
{
	static const string PROBE_ID = "e05";
	static const string SEGMENT_RESET = "reset";
	static const string SEGMENT_SETTLE = "settle";
	static const string SEGMENT_DRIVE = "drive";
	static const string SEGMENT_STOP = "stop";
	static const string SEGMENT_STATIC = "static";
	//! A stop that has not reached < 1 km/h after this long is reset anyway [s]
	static const float STOP_MAX_S = 20;
	//! Speed below which the car counts as stopped [km/h]
	static const float STOPPED_KMH = 1;

	// === RIG ===
	[Attribute("0 1.25 0.3", UIWidgets.Coords, "Camera mount in DC_Ego's local frame [m]: x right, y up, z forward [unverified: tune from the E-04 bounds and the first frames]", category: "Rig")]
	protected vector m_vMountPos;

	[Attribute("-4", UIWidgets.Slider, "Camera pitch [deg], negative = down (r2 3.4: -3 to -5)", "-30 30 0.5", category: "Rig")]
	protected float m_fMountPitchDeg;

	[Attribute("50", UIWidgets.Slider, "Vertical field of view [deg] (r2 3.4)", "10 120 1", category: "Rig")]
	protected float m_fVerticalFovDeg;

	[Attribute("0.1", UIWidgets.Slider, "Near clip plane [m]", "0.01 10 0.01", category: "Rig")]
	protected float m_fNearPlaneM;

	[Attribute("3000", UIWidgets.Slider, "Far clip plane [m]", "100 20000 10", category: "Rig")]
	protected float m_fFarPlaneM;

	// === CAPTURE ===
	[Attribute("10", UIWidgets.Slider, "Image slot rate [Hz] (world-time grid, one request in flight)", "1 60 1", category: "Capture")]
	protected float m_fImageHz;

	[Attribute("600", UIWidgets.Slider, "Capture duration [s of world time]; the probe timeout is this plus 120 s", "10 7200 10", category: "Capture")]
	protected float m_fDurationS;

	[Attribute("2000", UIWidgets.Slider, "A request whose file has not appeared after this long is resolved as timeout [ms of real time]", "100 60000 100", category: "Capture")]
	protected int m_iInFlightTimeoutMs;

	// === FIDUCIAL ===
	[Attribute("1", UIWidgets.CheckBox, "Draw fiducial v2 (needed for E-06)", category: "Fiducial")]
	protected bool m_bFiducial;

	[Attribute("32", UIWidgets.Slider, "Fiducial patch size [physical px]", "8 128 1", category: "Fiducial")]
	protected int m_iPatchPx;

	[Attribute("16", UIWidgets.Slider, "Fiducial margin from the left and right image edges [physical px]", "0 256 1", category: "Fiducial")]
	protected int m_iMarginPx;

	[Attribute("8", UIWidgets.Slider, "Vertical gap between fiducial patches [physical px]", "0 64 1", category: "Fiducial")]
	protected int m_iGapPx;

	// === DRIVE ===
	[Attribute("1", UIWidgets.CheckBox, "Shuttle along the test area while capturing (reset, settle, drive, stop); off = the car is not commanded (segment static)", category: "Drive")]
	protected bool m_bDrive;

	[Attribute("30", UIWidgets.Slider, "Drive speed [km/h] (probe speed hold, steering 0)", "5 120 1", category: "Drive")]
	protected float m_fDriveKmh;

	[Attribute("0.08", UIWidgets.Slider, "Probe speed hold: proportional gain [throttle per km/h] (as DC_ProbeSequences)", "0 1 0.001", category: "Drive")]
	protected float m_fSpeedKp;

	[Attribute("0.02", UIWidgets.Slider, "Probe speed hold: integral gain [throttle per km/h s] (as DC_ProbeSequences)", "0 1 0.001", category: "Drive")]
	protected float m_fSpeedKi;

	[Attribute("0.5", UIWidgets.Slider, "Probe speed hold: brake per unit of negative PI output (as DC_ProbeSequences)", "0 2 0.01", category: "Drive")]
	protected float m_fBrakeScale;

	// === AREA (same names and defaults as DC_ProbeSequences) ===
	[Attribute("659.6 0 2572.4", UIWidgets.Coords, "Start of the test area centre line (x, z used) [engine road dump E-08a 2026-10-22: Arland main runway (road 83, 60 m x 1217 m, y 38.9-39.2), 58 m in from its west end]", category: "Area")]
	protected vector m_vAreaStart;

	[Attribute("75.11", UIWidgets.Slider, "Heading of the test area centre line [deg], 0 = +Z, clockwise seen from above", "-180 180 0.01", category: "Area")]
	protected float m_fAreaHeadingDeg;

	[Attribute("1100", UIWidgets.Slider, "Length of the test area [m]", "50 5000 1", category: "Area")]
	protected float m_fAreaLengthM;

	[Attribute("25", UIWidgets.Slider, "Half width of the test area [m]", "2 200 0.5", category: "Area")]
	protected float m_fAreaHalfWidthM;

	[Attribute("3", UIWidgets.Slider, "Safety margin inside the test area [m]", "0 50 0.5", category: "Area")]
	protected float m_fAreaMarginM;

	[Attribute("0.3", UIWidgets.Slider, "Teleport height above GetSurfaceY [m]", "0 5 0.05", category: "Area")]
	protected float m_fSpawnLiftM;

	[Attribute("3", UIWidgets.Slider, "Settle time after a teleport, brake and handbrake on [s]", "0 30 0.5", category: "Area")]
	protected float m_fSettleS;

	// === STATE ===
	protected bool m_bRunning;
	protected bool m_bDone;
	protected DC_ProbeContext m_Ctx;
	protected DC_ProbeRunner m_Runner;
	protected IEntity m_Ego;
	protected ref DC_RigR1 m_Rig;
	protected ref DC_Fiducial m_Fiducial;
	protected ref DC_ImageSlotScheduler m_Scheduler;
	protected ref DC_ProbeActuator m_Actuator;
	protected ref DC_ProbeArea m_Area;
	protected DC_ProbeCsv m_FramesCsv;
	protected ref DC_ProbeRow m_FrameRow = new DC_ProbeRow();
	protected string m_sSegment;
	protected float m_fSegmentStartMs;
	protected float m_fStartWorldMs;
	protected int m_iShuttles;
	protected int m_iFrames;
	protected bool m_bTeleportPending;		//!< the reset segment waits for the teleport of the next OnProbeFrame
	protected float m_fStartAlongM;			//!< every reset puts the car this far along the centre line (2 x margin)

	//------------------------------------------------------------------------------------------------
	override void GetProbeIds(notnull array<string> outIds)
	{
		outIds.Insert(PROBE_ID);
	}

	//------------------------------------------------------------------------------------------------
	override bool IsConcurrent(string probeId)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	override bool NeedsEgo(string probeId)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Physics steps drive the shuttle; a static capture needs none
	override bool NeedsSteps(string probeId)
	{
		return m_bDrive;
	}

	//------------------------------------------------------------------------------------------------
	override float GetProbeTimeoutS(string probeId)
	{
		return m_fDurationS + 120;
	}

	//------------------------------------------------------------------------------------------------
	override void GetProbeParams(string probeId, notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("mount_pos=" + DC_RigR1.FormatVector(m_vMountPos, 3));
		outKeyValues.Insert("mount_pitch_deg=" + m_fMountPitchDeg.ToString(-1, 3));
		outKeyValues.Insert("vertical_fov_deg=" + m_fVerticalFovDeg.ToString(-1, 3));
		outKeyValues.Insert("near_plane_m=" + m_fNearPlaneM.ToString(-1, 3));
		outKeyValues.Insert("far_plane_m=" + m_fFarPlaneM.ToString(-1, 3));
		outKeyValues.Insert("image_hz=" + m_fImageHz.ToString(-1, 3));
		outKeyValues.Insert("duration_s=" + m_fDurationS.ToString(-1, 3));
		outKeyValues.Insert("inflight_timeout_ms=" + m_iInFlightTimeoutMs.ToString());
		outKeyValues.Insert("fiducial=" + BoolText(m_bFiducial));
		outKeyValues.Insert("patch_px=" + m_iPatchPx.ToString());
		outKeyValues.Insert("margin_px=" + m_iMarginPx.ToString());
		outKeyValues.Insert("gap_px=" + m_iGapPx.ToString());
		outKeyValues.Insert("drive=" + BoolText(m_bDrive));
		outKeyValues.Insert("drive_kmh=" + m_fDriveKmh.ToString(-1, 3));
		outKeyValues.Insert("speed_kp=" + m_fSpeedKp.ToString(-1, 4));
		outKeyValues.Insert("speed_ki=" + m_fSpeedKi.ToString(-1, 4));
		outKeyValues.Insert("brake_scale=" + m_fBrakeScale.ToString(-1, 4));
		outKeyValues.Insert("area_start=" + DC_RigR1.FormatVector(m_vAreaStart, 3));
		outKeyValues.Insert("area_heading_deg=" + m_fAreaHeadingDeg.ToString(-1, 3));
		outKeyValues.Insert("area_length_m=" + m_fAreaLengthM.ToString(-1, 3));
		outKeyValues.Insert("area_half_width_m=" + m_fAreaHalfWidthM.ToString(-1, 3));
		outKeyValues.Insert("area_margin_m=" + m_fAreaMarginM.ToString(-1, 3));
		outKeyValues.Insert("spawn_lift_m=" + m_fSpawnLiftM.ToString(-1, 3));
		outKeyValues.Insert("settle_s=" + m_fSettleS.ToString(-1, 3));
	}

	//------------------------------------------------------------------------------------------------
	override bool OnProbeStart(string probeId, notnull DC_ProbeContext ctx)
	{
		m_Ctx = ctx;
		m_Runner = ctx.GetRunner();
		m_Ego = ctx.GetEgo();
		m_bDone = false;
		m_iShuttles = 0;
		m_iFrames = 0;
		m_bTeleportPending = false;
		if (!m_Runner || !m_Ego)
		{
			ctx.Error("no runner or no ego - not started");
			return false;
		}

		m_Area = new DC_ProbeArea();
		m_Area.Set(m_vAreaStart, m_fAreaHeadingDeg, m_fAreaLengthM, m_fAreaHalfWidthM);

		// The shuttle starts where its own drive check (Contains with the margin) holds, as DC_ProbeSequences does
		m_fStartAlongM = 2 * m_fAreaMarginM;
		if (m_bDrive && !m_Area.Contains(m_Area.PointAt(m_fStartAlongM, 0), m_fAreaMarginM))
		{
			ctx.Error("shuttle start " + m_fStartAlongM.ToString(-1, 1) + " m along the area is not inside it with the margin " + m_fAreaMarginM.ToString(-1, 1) + " m (length " + m_fAreaLengthM.ToString(-1, 1) + " m) - not started");
			ReleaseAll();
			return false;
		}

		// The actuator also gives the speed column; it is required only when the probe drives.
		m_Actuator = new DC_ProbeActuator();
		if (!m_Actuator.Init(m_Ego))
		{
			if (m_bDrive)
			{
				ctx.Error("actuator: " + m_Actuator.GetFailure() + " - not started");
				ReleaseAll();
				return false;
			}

			ctx.Warn("actuator: " + m_Actuator.GetFailure() + " - veh_speed_kmh stays empty (static capture)");
			m_Actuator = null;
		}

		m_Rig = new DC_RigR1();
		if (!m_Rig.Attach(m_Ego, m_vMountPos, m_fMountPitchDeg, m_fVerticalFovDeg, m_fNearPlaneM, m_fFarPlaneM))
		{
			ctx.Error("rig " + DC_RigR1.RIG_NAME + ": " + m_Rig.GetFailure() + " - not started");
			ReleaseAll();
			return false;
		}

		if (m_bFiducial)
		{
			m_Fiducial = new DC_Fiducial();
			if (!m_Fiducial.Create(m_iPatchPx, m_iMarginPx, m_iGapPx))
			{
				ctx.Error("fiducial: " + m_Fiducial.GetFailure() + " - not started");
				ReleaseAll();
				return false;
			}
		}

		string frameDir = ctx.MakeSubdir("frames");
		if (frameDir.IsEmpty())
		{
			ctx.Error("cannot create the frames directory - not started");
			ReleaseAll();
			return false;
		}

		array<string> requestColumns = {};
		DC_ImageSlotScheduler.AddRequestColumns(requestColumns);
		DC_ProbeCsv requests = ctx.OpenCsv("e05_requests", requestColumns);

		array<string> frameColumns = {};
		AddFrameColumns(frameColumns);
		m_FramesCsv = ctx.OpenCsv("e05_frames", frameColumns);

		if (!requests || !m_FramesCsv)
		{
			ctx.Error("cannot open e05_requests.csv or e05_frames.csv - not started");
			ReleaseAll();
			return false;
		}

		m_fStartWorldMs = m_Runner.GetWorldTimeMs();
		m_Scheduler = new DC_ImageSlotScheduler();
		m_Scheduler.Init(frameDir, m_fImageHz, m_fStartWorldMs, m_iInFlightTimeoutMs, requests);

		array<string> header = {};
		m_Rig.AppendHeaderLines(header);
		if (m_Fiducial)
			m_Fiducial.AppendHeaderLines(header);
		else
			header.Insert("fiducial.created=0");

		AppendAreaLines(header, m_Area);
		header.Insert("area.start_along_m=" + m_fStartAlongM.ToString(-1, 3));
		header.Insert("capture.frames_dir=" + frameDir);
		header.Insert("capture.request_name=f_<slot_id:08d> (no extension)");
		header.Insert("capture.start_world_ms=" + m_fStartWorldMs.ToString(-1, 3));
		if (!ctx.AppendHeaderLines(header))
			ctx.Warn("could not append the rig, fiducial and area lines to run_header.txt");

		if (m_bDrive)
		{
			m_Actuator.SetSpeedGains(m_fSpeedKp, m_fSpeedKi, m_fBrakeScale);
			m_Actuator.StartModeA();
			EnterReset();
		}
		else
		{
			EnterSegment(SEGMENT_STATIC);
		}

		m_bRunning = true;
		float vFovRead = m_Rig.GetVerticalFovRead();
		ctx.Log("rig " + DC_RigR1.RIG_NAME + " current, vfov " + vFovRead.ToString(-1, 2) + ", " + m_fImageHz.ToString(-1, 1) + " Hz for " + m_fDurationS.ToString(-1, 0) + " s, drive " + BoolText(m_bDrive));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (!m_bRunning || !m_bDrive || !m_Actuator)
			return;

		m_Actuator.ApplyStep(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbePostSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (!m_bRunning || !m_bDrive || !m_Actuator)
			return;

		m_Actuator.ReadPost();
		UpdateShuttle();
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeFrame(string probeId, float timeSlice, int renderFrame)
	{
		if (!m_bRunning)
			return;

		if (m_Fiducial)
			m_Fiducial.Show(renderFrame);

		bool camCurrent = m_Rig.IsCurrent();
		bool camAttached = m_Rig.IsAttached();
		float worldMs = m_Runner.GetWorldTimeMs();

		// No slot at or after the deadline; what is in flight resolves with the camera and the bands still in place
		if (worldMs - m_fStartWorldMs >= m_fDurationS * 1000)
		{
			bool firstStop = m_Scheduler.StopIssuing();
			int inFlight = m_Scheduler.GetInFlight();
			if (firstStop && m_Ctx)
				m_Ctx.Log("capture time over: no new request, draining " + inFlight.ToString() + " in flight");
		}

		m_Scheduler.OnFrame(worldMs, renderFrame, m_Runner.GetPhysicsStep(), m_Runner.GetTickMs(), camCurrent, camAttached);

		VehicleWheeledSimulation simulation;
		if (m_Actuator)
			simulation = m_Actuator.GetSimulation();

		m_FrameRow.Clear();
		AddFrameValues(m_FrameRow, m_Runner, timeSlice, renderFrame, camCurrent, camAttached, m_Rig, m_Fiducial, m_Ego, simulation, m_Scheduler.GetInFlight(), m_sSegment);
		m_FramesCsv.WriteRow(m_FrameRow);
		m_iFrames++;

		// The shuttle's teleport runs here, outside the physics step; the reset segment waits for it
		if (m_bTeleportPending)
			TeleportToStart();

		if (m_Scheduler.IsDrained())
			m_bDone = true;
	}

	//------------------------------------------------------------------------------------------------
	override bool IsProbeDone(string probeId)
	{
		return m_bDone;
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeStop(string probeId, string reason)
	{
		if (!m_bRunning)
			return;

		m_bRunning = false;
		if (reason == DC_ProbeRunner.REASON_DELETE)
		{
			// World teardown: no camera, widget, setting or entity calls; the camera dies with the world.
			if (m_Scheduler)
				m_Scheduler.Abandon();

			ReleaseRefs();
			return;
		}

		m_Scheduler.Finish();
		if (m_Ctx)
			m_Ctx.Log(DescribeScheduler(m_Scheduler) + ", frames " + m_iFrames.ToString() + ", shuttles " + m_iShuttles.ToString() + ", reason " + reason);

		ParkVehicle();
		ReleaseAll();
	}

	//------------------------------------------------------------------------------------------------
	//! \return one log line with the scheduler counters
	static string DescribeScheduler(notnull DC_ImageSlotScheduler scheduler)
	{
		int issued = scheduler.GetIssued();
		int done = scheduler.GetDone();
		int skipped = scheduler.GetSkipped();
		int late = scheduler.GetSkippedLate();
		int timedOut = scheduler.GetTimedOut();
		int makeFailed = scheduler.GetMakeFailed();
		return "images: issued " + issued.ToString() + ", done " + done.ToString() + ", skipped " + skipped.ToString() + " (late " + late.ToString() + "), timed out " + timedOut.ToString() + ", make failed " + makeFailed.ToString();
	}

	//------------------------------------------------------------------------------------------------
	//! Shuttle state machine, once per physics step after ReadPost
	protected void UpdateShuttle()
	{
		float nowMs = m_Runner.GetWorldTimeMs();
		float segmentS = (nowMs - m_fSegmentStartMs) / 1000;

		if (m_sSegment == SEGMENT_RESET)
		{
			if (m_bTeleportPending)
				return;

			EnterSegment(SEGMENT_SETTLE);
			return;
		}

		if (m_sSegment == SEGMENT_SETTLE)
		{
			if (segmentS < m_fSettleS)
				return;

			if (!m_Actuator.IsEngineOn())
				m_Actuator.StartModeA();

			m_Actuator.SetCommand(0, 0, 0, 1, -1, false);
			m_Actuator.SetSpeedHold(m_fDriveKmh, 0);
			EnterSegment(SEGMENT_DRIVE);
			return;
		}

		if (m_sSegment == SEGMENT_DRIVE)
		{
			// Stop with two seconds of driving to spare before the far end of the area (minus its margin)
			float reserveM = 2 * m_fDriveKmh / 3.6;
			vector position = m_Ego.GetOrigin();
			bool atEnd = m_Area.AlongM(position) >= m_fAreaLengthM - m_fAreaMarginM - reserveM;
			if (atEnd || !m_Area.Contains(position, m_fAreaMarginM))
			{
				m_Actuator.SetCommand(0, 0, 1, 1, -1, false);
				EnterSegment(SEGMENT_STOP);
			}

			return;
		}

		if (m_sSegment == SEGMENT_STOP)
		{
			if (m_Actuator.GetSpeedKmh() < STOPPED_KMH || segmentS >= STOP_MAX_S)
			{
				m_iShuttles++;
				EnterReset();
			}
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Starts the reset segment; the teleport follows in the next OnProbeFrame (never inside a physics step)
	protected void EnterReset()
	{
		m_bTeleportPending = true;
		EnterSegment(SEGMENT_RESET);
	}

	//------------------------------------------------------------------------------------------------
	//! Called from OnProbeFrame: teleports the car to the start point m_fStartAlongM along the centre line
	//! (DC_ProbeActuator.Teleport also commands brake 1 and the handbrake) and lets the reset segment end
	protected void TeleportToStart()
	{
		m_bTeleportPending = false;
		vector start = m_Area.PointAt(m_fStartAlongM, 0);
		if (!m_Actuator.Teleport(start, m_fAreaHeadingDeg, m_fSpawnLiftM) && m_Ctx)
			m_Ctx.Warn("teleport to the area start failed");
	}

	//------------------------------------------------------------------------------------------------
	protected void EnterSegment(string segment)
	{
		m_sSegment = segment;
		m_fSegmentStartMs = m_Runner.GetWorldTimeMs();
	}

	//------------------------------------------------------------------------------------------------
	//! Brake and handbrake, applied at once (DC_ProbeActuator.Park): no physics step of this probe follows, and a
	//! driverless car keeps the last setter values
	protected void ParkVehicle()
	{
		if (!m_bDrive || !m_Actuator)
			return;

		m_Actuator.Park();
	}

	//------------------------------------------------------------------------------------------------
	//! Removes the bands, gives the camera back and drops every reference. Not for world teardown.
	protected void ReleaseAll()
	{
		if (m_Fiducial)
			m_Fiducial.Destroy();

		if (m_Rig)
			m_Rig.Detach();

		ReleaseRefs();
	}

	//------------------------------------------------------------------------------------------------
	protected void ReleaseRefs()
	{
		m_Fiducial = null;
		m_Rig = null;
		m_Scheduler = null;
		m_Actuator = null;
		m_Area = null;
		m_FramesCsv = null;
		m_Ego = null;
		m_Runner = null;
		m_Ctx = null;
	}

	//------------------------------------------------------------------------------------------------
	//! Appends the area.* lines of the test area and its measured relief (10 m grid) to lines
	static void AppendAreaLines(notnull array<string> lines, notnull DC_ProbeArea area)
	{
		area.AppendHeaderLines(lines, "area");
		BaseWorld world = GetGame().GetWorld();
		if (!world)
			return;

		float minY;
		float maxY;
		float relief = area.MeasureRelief(world, 10, minY, maxY);
		lines.Insert("area.relief_m=" + relief.ToString(-1, 3));
		lines.Insert("area.min_y=" + minY.ToString(-1, 3));
		lines.Insert("area.max_y=" + maxY.ToString(-1, 3));
	}

	//------------------------------------------------------------------------------------------------
	//! The columns of e05_frames.csv, also the first columns of e06a_frames.csv (the dc-2.0 hf prototype)
	static void AddFrameColumns(notnull array<string> columns)
	{
		columns.Insert("render_frame");
		columns.Insert("world_time_ms");
		columns.Insert("tick_ms");
		columns.Insert("frame_timeslice_s");
		columns.Insert("fps");
		columns.Insert("fps_avg");
		columns.Insert("last_physics_step");
		columns.Insert("paused");
		columns.Insert("cam_current");
		columns.Insert("cam_attached");
		columns.Insert("fiducial_code");
		DC_ProbeCsv.AddMat34Columns(columns, "cam");
		DC_ProbeCsv.AddMat34Columns(columns, "veh");
		columns.Insert("veh_speed_kmh");
		columns.Insert("inflight");
		columns.Insert("segment");
	}

	//------------------------------------------------------------------------------------------------
	//! Appends the values of AddFrameColumns for this render frame. fps = 1 / frame time slice (empty when 0),
	//! fps_avg = System.GetFPS(); fiducial_code empty without bands, veh_speed_kmh empty without a simulation.
	static void AddFrameValues(notnull DC_ProbeRow row, notnull DC_ProbeRunner runner, float timeSlice, int renderFrame, bool camCurrent, bool camAttached, DC_RigR1 rig, DC_Fiducial fiducial, IEntity ego, VehicleWheeledSimulation simulation, int inFlight, string segment)
	{
		row.AddInt(renderFrame);
		row.AddFloat(runner.GetWorldTimeMs(), 3);
		row.AddInt(runner.GetTickMs());
		row.AddFloat(timeSlice, 6);
		if (timeSlice > 0)
			row.AddFloat(1.0 / timeSlice, 3);
		else
			row.AddEmpty();

		row.AddFloat(System.GetFPS(), 3);
		row.AddInt(runner.GetPhysicsStep());
		row.AddBool(runner.IsPaused());
		row.AddBool(camCurrent);
		row.AddBool(camAttached);
		if (fiducial && fiducial.IsCreated())
			row.AddInt(fiducial.GetLastCode());
		else
			row.AddEmpty();

		vector camera[4];
		if (rig)
			rig.GetCameraTransform(camera);

		row.AddMat34(camera, 6, 3);

		vector vehicle[4];
		if (ego)
			ego.GetWorldTransform(vehicle);

		row.AddMat34(vehicle, 6, 3);

		if (simulation)
			row.AddFloat(simulation.GetSpeedKmh(), 3);
		else
			row.AddEmpty();

		row.AddInt(inFlight);
		row.AddString(segment);
	}

	//------------------------------------------------------------------------------------------------
	protected static string BoolText(bool value)
	{
		if (value)
			return "1";

		return "0";
	}
}
