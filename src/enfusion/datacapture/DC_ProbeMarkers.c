// ============================================================================
// DC_ProbeMarkers - T0-ENG probe E-06A: surveyed markers and reprojection data (DD-4-probe, r2 3.5)
// ============================================================================
//
// Probe id "e06a" (analysis: tools/validator/e06a_reproject.py). Exclusive, needs DC_Ego and a physics-step source.
// Uses rig R1, fiducial v2 and the image slot scheduler of DC_ProbeImages.c, and the actuator and test area of
// DC_ProbeSequences.c.
//
// WHAT IT DOES
//   - OnProbeStart: places the markers once (m_eMarkerKind): a straight set, m_fMarkerLateralM to each side of the
//     area centre line every m_fStraightStepM from m_fStraightFromM to m_fStraightToM, heights alternating through
//     m_sMarkerHeights; and a ring set of m_iRingCount markers on a circle of m_fRingRadiusM around the centre line at
//     m_fRingAlongM, m_fRingHeightM up. Heights are above BaseWorld.GetSurfaceY. Writes e06a_markers.csv, attaches
//     the rig, creates the fiducial bands (always on), opens the CSVs, saves VideoUserSettings.MaxFps and appends the
//     rig.*, fiducial.*, area.*, markers.* and fps.* lines to the run header;
//   - for each value of m_sFpsPhases (one "FPS phase"): sets MaxFps (BaseContainer.Set + UserSettingsChanged, never
//     SaveUserSettings), then the segments
//       reset   teleport to 2 x m_fAreaMarginM along the centre line (inside the margin-inset area, as
//               DC_ProbeSequences), heading along the area
//       settle  brake and handbrake for m_fSettleS
//       slalom  speed hold m_fSlalomKmh, steer m_fSlalomAmplitude * sin(2 pi t / m_fSlalomPeriodS), m_fSlalomS long,
//               images on the m_fImageHz grid with one request in flight
//       backlog the same slalom for m_fBacklogS with a screenshot request every frame (no in-flight limit)
//       brake   brake 1 from the current speed to a stop (images on the grid again)
//       reset   teleport to m_fRingRadiusM + RING_APPROACH_M before the ring centre, heading along the area
//       settle
//       curve   speed hold m_fCurveKmh, steer m_fCurveSteer for m_fCurveS or until the area edge
//     and at the end of the phase checks whether a prefab marker moved (> MOVED_M), writes e06a_markers_end.csv rows
//     and restores MaxFps. A slalom, backlog or curve that leaves the area (minus m_fAreaMarginM) goes on with the
//     next segment; its row says aborted_area. The teleports run in OnProbeFrame, never inside a physics step (the
//     rule of DC_ProbeSequences): a reset segment waits for its teleport;
//   - after the last FPS phase no new slot is issued and the requests in flight are drained with the camera and
//     the bands kept (DC_ImageSlotScheduler.StopIssuing); then the probe is done;
//   - every render frame: DC_Fiducial.Show, the scheduler, one e06a_frames.csv row and, on frames that issued a
//     request, e06a_proj.csv rows: WorkspaceWidget.ProjWorldToScreenNative of up to m_iProjMarkersPerFrame markers
//     within m_fProjMaxDistM in front of the camera (the engine's own projection, to validate the Python camera
//     model);
//   - OnProbeStop restores MaxFps, removes the markers and the bands, parks the car and gives the camera back.
//
// Debug-shape markers (Shape.CreateSphere, the default) have no collision. Prefab markers (the fallback when debug
// shapes do not show in screenshots) do: the straight set sits beside the slalom line, and a marker that moved is
// logged and excluded by the analysis. A prefab marker's logged position is the centre of its local bounds.
//
// Failure handling: OnProbeStart returns false and undoes what it did when the straight or the ring start point is
// not inside the margin-inset area, or when the actuator, the markers, the rig, the bands, frames/ or a CSV fail.
// Reason "delete" (world teardown) touches no camera, widget, setting or entity: it only drops references (MaxFps
// then stays at the phase value: the engine settings are not saved by this probe).
// ============================================================================

//------------------------------------------------------------------------------------------------
//! How the markers are drawn
enum DC_EProbeMarkerKind
{
	SHAPE,		//!< debug sphere (Shape.CreateSphere), exact centre, any colour
	PREFAB		//!< vanilla prefab m_sMarkerPrefab (fallback if debug shapes are not in screenshots)
}

//------------------------------------------------------------------------------------------------
//! One surveyed marker
class DC_ProbeMarker
{
	int m_iId;
	string m_sSet;				//!< "straight" or "ring"
	vector m_vCentre;			//!< logged position: sphere centre, or the world centre of the prefab's local bounds
	vector m_vSpawnOrigin;		//!< prefab origin at spawn (moved check)
	float m_fRadiusM;
	IEntity m_Entity;			//!< prefab markers only
	ref Shape m_Shape;			//!< debug-shape markers only; the sphere is drawn while this reference lives
	bool m_bMoved;
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "T0-ENG probe E-06A: surveyed markers, slalom and braking at two FPS settings with screenshot backlog, for reprojection alignment")]
class DC_ProbeMarkersClass : DC_ProbeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeMarkers : DC_ProbeComponent
{
	static const string PROBE_ID = "e06a";
	static const string SEGMENT_RESET = "reset";
	static const string SEGMENT_SETTLE = "settle";
	static const string SEGMENT_SLALOM = "slalom";
	static const string SEGMENT_BACKLOG = "backlog";
	static const string SEGMENT_BRAKE = "brake";
	static const string SEGMENT_CURVE = "curve";

	//! Stages of one FPS phase, in order
	static const int STAGE_RESET_STRAIGHT = 0;
	static const int STAGE_SETTLE_STRAIGHT = 1;
	static const int STAGE_SLALOM = 2;
	static const int STAGE_BACKLOG = 3;
	static const int STAGE_BRAKE = 4;
	static const int STAGE_RESET_RING = 5;
	static const int STAGE_SETTLE_RING = 6;
	static const int STAGE_CURVE = 7;
	static const int STAGE_DONE = 8;

	//! The curve starts this far before the near edge of the ring [m] (PROBE_DESIGN.md 5.5)
	static const float RING_APPROACH_M = 15;
	//! A prefab marker that moved more than this is excluded by the analysis [m]
	static const float MOVED_M = 0.1;
	//! A brake that has not reached < STOPPED_KMH after this long ends anyway (result timeout) [s]
	static const float BRAKE_MAX_S = 20;
	static const float STOPPED_KMH = 1;
	//! Brake result low_speed when the brake starts this much below m_fBrakeFromKmh [km/h]
	static const float BRAKE_SPEED_TOL_KMH = 5;
	static const string VIDEO_MODULE = "VideoUserSettings";
	static const string MAX_FPS_KEY = "MaxFps";

	// === RIG (as DC_ProbeImages) ===
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

	// === FIDUCIAL (always on) ===
	[Attribute("32", UIWidgets.Slider, "Fiducial patch size [physical px]", "8 128 1", category: "Fiducial")]
	protected int m_iPatchPx;

	[Attribute("16", UIWidgets.Slider, "Fiducial margin from the left and right image edges [physical px]", "0 256 1", category: "Fiducial")]
	protected int m_iMarginPx;

	[Attribute("8", UIWidgets.Slider, "Vertical gap between fiducial patches [physical px]", "0 64 1", category: "Fiducial")]
	protected int m_iGapPx;

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

	// === MARKERS ===
	[Attribute("0", UIWidgets.ComboBox, "Marker kind: SHAPE = debug sphere (exact centre, no collision), PREFAB = m_sMarkerPrefab (fallback when debug shapes are not in screenshots)", "", ParamEnumArray.FromEnum(DC_EProbeMarkerKind), category: "Markers")]
	protected DC_EProbeMarkerKind m_eMarkerKind;

	[Attribute("{25C8B889A777399D}Prefabs/Props/Infrastructure/ConeTraffic/ConeTraffic_01_red.et", UIWidgets.ResourcePickerThumbnail, "Marker prefab for the PREFAB kind (the analysis then needs --color of the prefab)", "et", category: "Markers")]
	protected ResourceName m_sMarkerPrefab;

	[Attribute("1 0 1", UIWidgets.Coords, "Debug-sphere colour, RGB 0..1 (magenta; the analysis default --color FF00FF)", category: "Markers")]
	protected vector m_vMarkerColor;

	[Attribute("0.25", UIWidgets.Slider, "Debug-sphere radius [m]", "0.05 5 0.01", category: "Markers")]
	protected float m_fMarkerRadiusM;

	[Attribute("20", UIWidgets.Slider, "Straight set: first marker pair, along the area [m]", "0 2000 1", category: "Markers")]
	protected float m_fStraightFromM;

	[Attribute("400", UIWidgets.Slider, "Straight set: last marker pair, along the area [m]", "0 5000 1", category: "Markers")]
	protected float m_fStraightToM;

	[Attribute("20", UIWidgets.Slider, "Straight set: spacing of the marker pairs [m]", "1 500 1", category: "Markers")]
	protected float m_fStraightStepM;

	[Attribute("10", UIWidgets.Slider, "Straight set: lateral offset of each marker from the centre line, both sides [m]", "1 100 0.5", category: "Markers")]
	protected float m_fMarkerLateralM;

	[Attribute("0.6,1.6", UIWidgets.EditBox, "Straight set: marker heights above ground [m], cycled pair by pair", category: "Markers")]
	protected string m_sMarkerHeights;

	[Attribute("700", UIWidgets.Slider, "Ring set: centre along the area [m]", "0 5000 1", category: "Markers")]
	protected float m_fRingAlongM;

	[Attribute("25", UIWidgets.Slider, "Ring set: radius [m]", "1 200 0.5", category: "Markers")]
	protected float m_fRingRadiusM;

	[Attribute("24", UIWidgets.Slider, "Ring set: number of markers", "0 128 1", category: "Markers")]
	protected int m_iRingCount;

	[Attribute("0.8", UIWidgets.Slider, "Ring set: marker height above ground [m]", "0 10 0.05", category: "Markers")]
	protected float m_fRingHeightM;

	// === DRIVE ===
	[Attribute("40", UIWidgets.Slider, "Slalom speed [km/h] (r2 3.5: 40)", "5 120 1", category: "Drive")]
	protected float m_fSlalomKmh;

	[Attribute("0.3", UIWidgets.Slider, "Slalom steering amplitude (r2 3.5: +-0.3)", "0 1 0.01", category: "Drive")]
	protected float m_fSlalomAmplitude;

	[Attribute("3", UIWidgets.Slider, "Slalom steering period [s]", "0.5 30 0.1", category: "Drive")]
	protected float m_fSlalomPeriodS;

	[Attribute("30", UIWidgets.Slider, "Slalom duration with images on the grid [s]", "1 300 1", category: "Drive")]
	protected float m_fSlalomS;

	[Attribute("10", UIWidgets.Slider, "Slalom continued with a screenshot request every frame (backlog) [s]", "0 120 1", category: "Drive")]
	protected float m_fBacklogS;

	[Attribute("40", UIWidgets.Slider, "Expected speed at the start of the hard brake [km/h]; the brake row says low_speed when it starts more than 5 km/h below", "5 120 1", category: "Drive")]
	protected float m_fBrakeFromKmh;

	[Attribute("25", UIWidgets.Slider, "Curve speed [km/h]", "5 120 1", category: "Drive")]
	protected float m_fCurveKmh;

	[Attribute("0.2", UIWidgets.Slider, "Curve steering (held)", "-1 1 0.01", category: "Drive")]
	protected float m_fCurveSteer;

	[Attribute("15", UIWidgets.Slider, "Curve duration [s] (ends earlier at the area edge)", "1 300 1", category: "Drive")]
	protected float m_fCurveS;

	[Attribute("0.08", UIWidgets.Slider, "Probe speed hold: proportional gain [throttle per km/h] (as DC_ProbeSequences)", "0 1 0.001", category: "Drive")]
	protected float m_fSpeedKp;

	[Attribute("0.02", UIWidgets.Slider, "Probe speed hold: integral gain [throttle per km/h s] (as DC_ProbeSequences)", "0 1 0.001", category: "Drive")]
	protected float m_fSpeedKi;

	[Attribute("0.5", UIWidgets.Slider, "Probe speed hold: brake per unit of negative PI output (as DC_ProbeSequences)", "0 2 0.01", category: "Drive")]
	protected float m_fBrakeScale;

	// === CAPTURE ===
	[Attribute("10", UIWidgets.Slider, "Image slot rate [Hz] outside the backlog segment (world-time grid, one request in flight)", "1 60 1", category: "Capture")]
	protected float m_fImageHz;

	[Attribute("2000", UIWidgets.Slider, "A request whose file has not appeared after this long is resolved as timeout [ms of real time]", "100 60000 100", category: "Capture")]
	protected int m_iInFlightTimeoutMs;

	[Attribute("60,30", UIWidgets.EditBox, "VideoUserSettings.MaxFps per FPS phase, in order (r2 3.5: 60 and 30)", category: "Capture")]
	protected string m_sFpsPhases;

	[Attribute("16", UIWidgets.Slider, "e06a_proj.csv: markers projected by the engine per request frame", "0 128 1", category: "Capture")]
	protected int m_iProjMarkersPerFrame;

	[Attribute("150", UIWidgets.Slider, "e06a_proj.csv: only markers within this distance in front of the camera [m]", "1 2000 1", category: "Capture")]
	protected float m_fProjMaxDistM;

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
	protected ref array<ref DC_ProbeMarker> m_aMarkers = {};
	protected ref array<int> m_aFpsPhases = {};
	protected DC_ProbeCsv m_FramesCsv;
	protected DC_ProbeCsv m_PhasesCsv;
	protected DC_ProbeCsv m_MarkersEndCsv;
	protected DC_ProbeCsv m_ProjCsv;
	protected ref DC_ProbeRow m_Row = new DC_ProbeRow();

	protected int m_iFpsPhase;					//!< index into m_aFpsPhases
	protected int m_iStage;
	protected string m_sSegment;
	protected string m_sSegmentResult;
	protected float m_fSegmentStartMs;
	protected int m_iSegmentStartFrame;
	protected float m_fSlalomStartMs;			//!< slalom time origin, shared by slalom and backlog
	protected int m_iPhaseRows;					//!< e06a_phases.csv rows written = next phase_idx

	protected bool m_bMaxFpsSaved;
	protected int m_iMaxFpsOriginal;
	protected bool m_bMaxFpsChanged;			//!< MaxFps currently differs from the saved value
	protected int m_iFrames;
	protected bool m_bTeleportPending;			//!< a reset segment waits for the teleport of the next OnProbeFrame
	protected float m_fTeleportAlongM;
	protected float m_fStartAlongM;				//!< start of the straight segments: 2 x m_fAreaMarginM along

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
	override bool NeedsSteps(string probeId)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override float GetProbeTimeoutS(string probeId)
	{
		return 1800;
	}

	//------------------------------------------------------------------------------------------------
	override void GetProbeParams(string probeId, notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("mount_pos=" + DC_RigR1.FormatVector(m_vMountPos, 3));
		outKeyValues.Insert("mount_pitch_deg=" + m_fMountPitchDeg.ToString(-1, 3));
		outKeyValues.Insert("vertical_fov_deg=" + m_fVerticalFovDeg.ToString(-1, 3));
		outKeyValues.Insert("near_plane_m=" + m_fNearPlaneM.ToString(-1, 3));
		outKeyValues.Insert("far_plane_m=" + m_fFarPlaneM.ToString(-1, 3));
		outKeyValues.Insert("patch_px=" + m_iPatchPx.ToString());
		outKeyValues.Insert("margin_px=" + m_iMarginPx.ToString());
		outKeyValues.Insert("gap_px=" + m_iGapPx.ToString());
		outKeyValues.Insert("area_start=" + DC_RigR1.FormatVector(m_vAreaStart, 3));
		outKeyValues.Insert("area_heading_deg=" + m_fAreaHeadingDeg.ToString(-1, 3));
		outKeyValues.Insert("area_length_m=" + m_fAreaLengthM.ToString(-1, 3));
		outKeyValues.Insert("area_half_width_m=" + m_fAreaHalfWidthM.ToString(-1, 3));
		outKeyValues.Insert("area_margin_m=" + m_fAreaMarginM.ToString(-1, 3));
		outKeyValues.Insert("spawn_lift_m=" + m_fSpawnLiftM.ToString(-1, 3));
		outKeyValues.Insert("settle_s=" + m_fSettleS.ToString(-1, 3));
		outKeyValues.Insert("marker_kind=" + typename.EnumToString(DC_EProbeMarkerKind, m_eMarkerKind));
		outKeyValues.Insert("marker_prefab=" + m_sMarkerPrefab);
		outKeyValues.Insert("marker_color=" + DC_RigR1.FormatVector(m_vMarkerColor, 3));
		outKeyValues.Insert("marker_radius_m=" + m_fMarkerRadiusM.ToString(-1, 3));
		outKeyValues.Insert("straight_from_m=" + m_fStraightFromM.ToString(-1, 3));
		outKeyValues.Insert("straight_to_m=" + m_fStraightToM.ToString(-1, 3));
		outKeyValues.Insert("straight_step_m=" + m_fStraightStepM.ToString(-1, 3));
		outKeyValues.Insert("marker_lateral_m=" + m_fMarkerLateralM.ToString(-1, 3));
		outKeyValues.Insert("marker_heights=" + m_sMarkerHeights);
		outKeyValues.Insert("ring_along_m=" + m_fRingAlongM.ToString(-1, 3));
		outKeyValues.Insert("ring_radius_m=" + m_fRingRadiusM.ToString(-1, 3));
		outKeyValues.Insert("ring_count=" + m_iRingCount.ToString());
		outKeyValues.Insert("ring_height_m=" + m_fRingHeightM.ToString(-1, 3));
		outKeyValues.Insert("slalom_kmh=" + m_fSlalomKmh.ToString(-1, 3));
		outKeyValues.Insert("slalom_amplitude=" + m_fSlalomAmplitude.ToString(-1, 4));
		outKeyValues.Insert("slalom_period_s=" + m_fSlalomPeriodS.ToString(-1, 3));
		outKeyValues.Insert("slalom_s=" + m_fSlalomS.ToString(-1, 3));
		outKeyValues.Insert("backlog_s=" + m_fBacklogS.ToString(-1, 3));
		outKeyValues.Insert("brake_from_kmh=" + m_fBrakeFromKmh.ToString(-1, 3));
		outKeyValues.Insert("curve_kmh=" + m_fCurveKmh.ToString(-1, 3));
		outKeyValues.Insert("curve_steer=" + m_fCurveSteer.ToString(-1, 4));
		outKeyValues.Insert("curve_s=" + m_fCurveS.ToString(-1, 3));
		outKeyValues.Insert("speed_kp=" + m_fSpeedKp.ToString(-1, 4));
		outKeyValues.Insert("speed_ki=" + m_fSpeedKi.ToString(-1, 4));
		outKeyValues.Insert("brake_scale=" + m_fBrakeScale.ToString(-1, 4));
		outKeyValues.Insert("image_hz=" + m_fImageHz.ToString(-1, 3));
		outKeyValues.Insert("inflight_timeout_ms=" + m_iInFlightTimeoutMs.ToString());
		outKeyValues.Insert("fps_phases=" + m_sFpsPhases);
		outKeyValues.Insert("proj_markers_per_frame=" + m_iProjMarkersPerFrame.ToString());
		outKeyValues.Insert("proj_max_dist_m=" + m_fProjMaxDistM.ToString(-1, 3));
	}

	//------------------------------------------------------------------------------------------------
	override bool OnProbeStart(string probeId, notnull DC_ProbeContext ctx)
	{
		m_Ctx = ctx;
		m_Runner = ctx.GetRunner();
		m_Ego = ctx.GetEgo();
		m_bDone = false;
		m_iFrames = 0;
		m_iPhaseRows = 0;
		m_iFpsPhase = 0;
		m_bTeleportPending = false;
		if (!m_Runner || !m_Ego)
		{
			ctx.Error("no runner or no ego - not started");
			return false;
		}

		if (!ParseFpsPhases())
		{
			ctx.Error("m_sFpsPhases '" + m_sFpsPhases + "' holds no FPS value > 0 - not started");
			return false;
		}

		m_Area = new DC_ProbeArea();
		m_Area.Set(m_vAreaStart, m_fAreaHeadingDeg, m_fAreaLengthM, m_fAreaHalfWidthM);

		// Both teleport targets must pass the segments' own area check (Contains with the margin)
		m_fStartAlongM = 2 * m_fAreaMarginM;
		string startFailure = CheckStartPoints();
		if (!startFailure.IsEmpty())
		{
			ctx.Error(startFailure + " - not started");
			ReleaseAll();
			return false;
		}

		m_Actuator = new DC_ProbeActuator();
		if (!m_Actuator.Init(m_Ego))
		{
			ctx.Error("actuator: " + m_Actuator.GetFailure() + " - not started");
			ReleaseAll();
			return false;
		}

		string failure = PlaceMarkers();
		if (!failure.IsEmpty())
		{
			ctx.Error("markers: " + failure + " - not started");
			ReleaseAll();
			return false;
		}

		m_Rig = new DC_RigR1();
		if (!m_Rig.Attach(m_Ego, m_vMountPos, m_fMountPitchDeg, m_fVerticalFovDeg, m_fNearPlaneM, m_fFarPlaneM))
		{
			ctx.Error("rig " + DC_RigR1.RIG_NAME + ": " + m_Rig.GetFailure() + " - not started");
			ReleaseAll();
			return false;
		}

		m_Fiducial = new DC_Fiducial();
		if (!m_Fiducial.Create(m_iPatchPx, m_iMarginPx, m_iGapPx))
		{
			ctx.Error("fiducial: " + m_Fiducial.GetFailure() + " - not started");
			ReleaseAll();
			return false;
		}

		string frameDir = ctx.MakeSubdir("frames");
		if (frameDir.IsEmpty() || !OpenFiles(ctx))
		{
			ctx.Error("cannot create frames/ or open the e06a CSVs - not started");
			ReleaseAll();
			return false;
		}

		DC_ProbeCsv requests = OpenRequests(ctx);
		if (!requests || !WriteMarkers(ctx))
		{
			ctx.Error("cannot write e06a_requests.csv or e06a_markers.csv - not started");
			ReleaseAll();
			return false;
		}

		SaveMaxFps();

		m_Scheduler = new DC_ImageSlotScheduler();
		m_Scheduler.Init(frameDir, m_fImageHz, m_Runner.GetWorldTimeMs(), m_iInFlightTimeoutMs, requests);

		array<string> header = {};
		m_Rig.AppendHeaderLines(header);
		m_Fiducial.AppendHeaderLines(header);
		DC_ProbeImages.AppendAreaLines(header, m_Area);
		float ringStartAlongM = GetRingStartAlongM();
		header.Insert("area.start_along_m=" + m_fStartAlongM.ToString(-1, 3));
		header.Insert("area.ring_start_along_m=" + ringStartAlongM.ToString(-1, 3));
		AppendMarkerLines(header);
		header.Insert("capture.frames_dir=" + frameDir);
		header.Insert("capture.request_name=f_<slot_id:08d> (no extension)");
		if (!ctx.AppendHeaderLines(header))
			ctx.Warn("could not append the rig, fiducial, area and marker lines to run_header.txt");

		m_Actuator.SetSpeedGains(m_fSpeedKp, m_fSpeedKi, m_fBrakeScale);
		m_Actuator.StartModeA();
		m_bRunning = true;
		StartFpsPhase();

		int markerCount = m_aMarkers.Count();
		int phaseCount = m_aFpsPhases.Count();
		ctx.Log("markers " + markerCount.ToString() + " (" + typename.EnumToString(DC_EProbeMarkerKind, m_eMarkerKind) + "), FPS phases " + phaseCount.ToString() + " (" + m_sFpsPhases + "), MaxFps saved " + BoolText(m_bMaxFpsSaved));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeSimulate(string probeId, float timeSlice, int physicsStep)
	{
		// After the last phase the park command (brake and handbrake) is still applied every step until the stop
		if (!m_bRunning)
			return;

		if (m_iStage == STAGE_SLALOM || m_iStage == STAGE_BACKLOG)
		{
			float t = (m_Runner.GetWorldTimeMs() - m_fSlalomStartMs) / 1000;
			float steer = m_fSlalomAmplitude * Math.Sin(Math.PI2 * t / m_fSlalomPeriodS);
			m_Actuator.SetSpeedHold(m_fSlalomKmh, steer);
		}

		m_Actuator.ApplyStep(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbePostSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (!m_bRunning)
			return;

		m_Actuator.ReadPost();
		if (m_iStage != STAGE_DONE)
			UpdateStage();
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeFrame(string probeId, float timeSlice, int renderFrame)
	{
		if (!m_bRunning)
			return;

		m_Fiducial.Show(renderFrame);

		// After the last FPS phase: no new slot; what is in flight resolves with the camera and the bands in place
		if (m_iStage == STAGE_DONE)
		{
			bool firstStop = m_Scheduler.StopIssuing();
			if (firstStop && m_Ctx)
				m_Ctx.Log("all FPS phases done: no new request, draining the requests in flight");
		}

		bool camCurrent = m_Rig.IsCurrent();
		bool camAttached = m_Rig.IsAttached();
		int issuedBefore = m_Scheduler.GetIssued();
		m_Scheduler.OnFrame(m_Runner.GetWorldTimeMs(), renderFrame, m_Runner.GetPhysicsStep(), m_Runner.GetTickMs(), camCurrent, camAttached);

		m_Row.Clear();
		DC_ProbeImages.AddFrameValues(m_Row, m_Runner, timeSlice, renderFrame, camCurrent, camAttached, m_Rig, m_Fiducial, m_Ego, m_Actuator.GetSimulation(), m_Scheduler.GetInFlight(), m_sSegment);
		m_Row.AddInt(CurrentFpsSetting());
		m_FramesCsv.WriteRow(m_Row);
		m_iFrames++;

		if (m_Scheduler.GetIssued() > issuedBefore)
			WriteProjections(renderFrame);

		// A requested teleport runs here, outside the physics step; the reset segment waits for it
		if (m_bTeleportPending)
			RunPendingTeleport();

		if (m_iStage == STAGE_DONE && m_Scheduler.IsDrained())
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
			// World teardown: no camera, widget, setting or entity calls; markers and camera die with the world.
			if (m_Scheduler)
				m_Scheduler.Abandon();

			foreach (DC_ProbeMarker marker : m_aMarkers)
			{
				marker.m_Entity = null;
			}

			m_aMarkers.Clear();
			ReleaseRefs();
			return;
		}

		if (m_iStage != STAGE_DONE)
			WritePhaseRow("stopped_" + reason);

		m_Scheduler.Finish();
		if (m_Ctx)
			m_Ctx.Log(DC_ProbeImages.DescribeScheduler(m_Scheduler) + ", frames " + m_iFrames.ToString() + ", reason " + reason);

		RestoreMaxFps();
		m_Actuator.Park();
		ReleaseAll();
	}

	//------------------------------------------------------------------------------------------------
	//! Stage machine, once per physics step after ReadPost
	protected void UpdateStage()
	{
		float segmentS = (m_Runner.GetWorldTimeMs() - m_fSegmentStartMs) / 1000;
		vector position = m_Ego.GetOrigin();
		bool inArea = m_Area.Contains(position, m_fAreaMarginM);

		if (m_iStage == STAGE_RESET_STRAIGHT)
		{
			if (m_bTeleportPending)
				return;

			EnterStage(STAGE_SETTLE_STRAIGHT, SEGMENT_SETTLE, "ok");
			return;
		}

		if (m_iStage == STAGE_SETTLE_STRAIGHT)
		{
			if (segmentS < m_fSettleS)
				return;

			StartDriving(m_fSlalomKmh, 0);
			m_fSlalomStartMs = m_Runner.GetWorldTimeMs();
			EnterStage(STAGE_SLALOM, SEGMENT_SLALOM, "ok");
			return;
		}

		if (m_iStage == STAGE_SLALOM)
		{
			if (!inArea)
			{
				EnterBrake("aborted_area");
				return;
			}

			if (segmentS < m_fSlalomS)
				return;

			if (m_fBacklogS <= 0)
			{
				EnterBrake("ok");
				return;
			}

			EnterStage(STAGE_BACKLOG, SEGMENT_BACKLOG, "ok");
			m_Scheduler.SetBacklog(true);
			return;
		}

		if (m_iStage == STAGE_BACKLOG)
		{
			if (inArea && segmentS < m_fBacklogS)
				return;

			m_Scheduler.SetBacklog(false);
			if (inArea)
				EnterBrake("ok");
			else
				EnterBrake("aborted_area");

			return;
		}

		if (m_iStage == STAGE_BRAKE)
		{
			bool stopped = m_Actuator.GetSpeedKmh() < STOPPED_KMH;
			if (!stopped && segmentS < BRAKE_MAX_S)
				return;

			if (!stopped)
				m_sSegmentResult = "timeout";

			TeleportTo(GetRingStartAlongM());
			EnterStage(STAGE_RESET_RING, SEGMENT_RESET, "ok");
			return;
		}

		if (m_iStage == STAGE_RESET_RING)
		{
			if (m_bTeleportPending)
				return;

			EnterStage(STAGE_SETTLE_RING, SEGMENT_SETTLE, "ok");
			return;
		}

		if (m_iStage == STAGE_SETTLE_RING)
		{
			if (segmentS < m_fSettleS)
				return;

			StartDriving(m_fCurveKmh, m_fCurveSteer);
			EnterStage(STAGE_CURVE, SEGMENT_CURVE, "ok");
			return;
		}

		if (m_iStage == STAGE_CURVE)
		{
			if (inArea && segmentS < m_fCurveS)
				return;

			if (!inArea)
				m_sSegmentResult = "aborted_area";

			EndFpsPhase();
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Releases brake and handbrake and holds targetKmh with a held steering value (the slalom updates it per step)
	protected void StartDriving(float targetKmh, float steer)
	{
		if (!m_Actuator.IsEngineOn())
			m_Actuator.StartModeA();

		m_Actuator.SetCommand(0, 0, 0, 1, -1, false);
		m_Actuator.SetSpeedHold(targetKmh, steer);
	}

	//------------------------------------------------------------------------------------------------
	//! Sets MaxFps for the current FPS phase and starts its first segment
	protected void StartFpsPhase()
	{
		ApplyMaxFps(CurrentFpsSetting());
		TeleportTo(m_fStartAlongM);
		m_sSegment = string.Empty;
		EnterStage(STAGE_RESET_STRAIGHT, SEGMENT_RESET, "ok");
	}

	//------------------------------------------------------------------------------------------------
	//! Closes the curve row, checks the markers, restores MaxFps, then the next FPS phase or done
	protected void EndFpsPhase()
	{
		WritePhaseRow(m_sSegmentResult);
		int phaseIdx = m_iPhaseRows - 1;
		CheckMarkers(phaseIdx);
		RestoreMaxFps();

		m_iFpsPhase++;
		if (m_iFpsPhase < m_aFpsPhases.Count())
		{
			StartFpsPhase();
			return;
		}

		m_Actuator.SetCommand(0, 0, 1, 1, -1, true);
		m_iStage = STAGE_DONE;
		m_sSegment = "done";
		UpdateExtraValues();
	}

	//------------------------------------------------------------------------------------------------
	protected void EnterBrake(string result)
	{
		m_sSegmentResult = result;
		float speed = m_Actuator.GetSpeedKmh();
		m_Actuator.SetCommand(0, 0, 1, 1, -1, false);
		string brakeResult = "ok";
		if (speed < m_fBrakeFromKmh - BRAKE_SPEED_TOL_KMH)
			brakeResult = "low_speed";

		EnterStage(STAGE_BRAKE, SEGMENT_BRAKE, brakeResult);
	}

	//------------------------------------------------------------------------------------------------
	//! Writes the row of the segment that ends (if any) and starts the next one
	protected void EnterStage(int stage, string segment, string result)
	{
		if (!m_sSegment.IsEmpty())
			WritePhaseRow(m_sSegmentResult);

		m_iStage = stage;
		m_sSegment = segment;
		m_sSegmentResult = result;
		m_fSegmentStartMs = m_Runner.GetWorldTimeMs();
		m_iSegmentStartFrame = m_Runner.GetRenderFrame();
		UpdateExtraValues();
	}

	//------------------------------------------------------------------------------------------------
	//! One e06a_phases.csv row for the current segment
	protected void WritePhaseRow(string result)
	{
		if (!m_PhasesCsv || m_sSegment.IsEmpty())
			return;

		m_Row.Clear();
		m_Row.AddInt(m_iPhaseRows);
		m_Row.AddInt(CurrentFpsSetting());
		m_Row.AddString(m_sSegment);
		m_Row.AddFloat(m_fSegmentStartMs, 3);
		m_Row.AddFloat(m_Runner.GetWorldTimeMs(), 3);
		m_Row.AddInt(m_iSegmentStartFrame);
		m_Row.AddInt(m_Runner.GetRenderFrame());
		m_Row.AddString(result);
		m_PhasesCsv.WriteRow(m_Row);
		m_iPhaseRows++;
	}

	//------------------------------------------------------------------------------------------------
	protected void UpdateExtraValues()
	{
		if (!m_Scheduler)
			return;

		array<string> extra = {};
		int fps = CurrentFpsSetting();
		extra.Insert(fps.ToString());
		extra.Insert(m_sSegment);
		m_Scheduler.SetExtraValues(extra);
	}

	//------------------------------------------------------------------------------------------------
	//! \return MaxFps of the current FPS phase, or the last one after the plan
	protected int CurrentFpsSetting()
	{
		if (m_aFpsPhases.IsEmpty())
			return 0;

		int index = m_iFpsPhase;
		if (index >= m_aFpsPhases.Count())
			index = m_aFpsPhases.Count() - 1;

		return m_aFpsPhases[index];
	}

	//------------------------------------------------------------------------------------------------
	//! \return along position of the curve start: m_fRingRadiusM + RING_APPROACH_M before the ring centre
	protected float GetRingStartAlongM()
	{
		return m_fRingAlongM - m_fRingRadiusM - RING_APPROACH_M;
	}

	//------------------------------------------------------------------------------------------------
	//! \return empty when the straight start (m_fStartAlongM) and the ring start lie inside the area shrunk by the
	//! margin (the check every driving segment makes), else the failure
	protected string CheckStartPoints()
	{
		float ringStart = GetRingStartAlongM();
		if (!m_Area.Contains(m_Area.PointAt(m_fStartAlongM, 0), m_fAreaMarginM))
			return "straight start " + m_fStartAlongM.ToString(-1, 1) + " m along the area is not inside it with the margin " + m_fAreaMarginM.ToString(-1, 1) + " m";

		if (!m_Area.Contains(m_Area.PointAt(ringStart, 0), m_fAreaMarginM))
			return "ring start " + ringStart.ToString(-1, 1) + " m along the area (m_fRingAlongM - m_fRingRadiusM - 15) is not inside it with the margin " + m_fAreaMarginM.ToString(-1, 1) + " m";

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! Requests a teleport to alongM on the centre line, heading along the area; the next OnProbeFrame runs it
	//! (never inside a physics step) and the reset segment waits for it
	protected void TeleportTo(float alongM)
	{
		m_fTeleportAlongM = alongM;
		m_bTeleportPending = true;
	}

	//------------------------------------------------------------------------------------------------
	//! Called from OnProbeFrame. DC_ProbeActuator.Teleport also commands brake 1 and the handbrake.
	protected void RunPendingTeleport()
	{
		m_bTeleportPending = false;
		if (!m_Actuator.Teleport(m_Area.PointAt(m_fTeleportAlongM, 0), m_fAreaHeadingDeg, m_fSpawnLiftM) && m_Ctx)
			m_Ctx.Warn("teleport to " + m_fTeleportAlongM.ToString(-1, 1) + " m along the area failed");
	}

	//------------------------------------------------------------------------------------------------
	//! \return false when m_sFpsPhases holds no value > 0
	protected bool ParseFpsPhases()
	{
		m_aFpsPhases.Clear();
		array<string> tokens = {};
		m_sFpsPhases.Split(",", tokens, true);
		foreach (string token : tokens)
		{
			string trimmed = token.Trim();
			int fps = trimmed.ToInt();
			if (fps > 0)
				m_aFpsPhases.Insert(fps);
		}

		return !m_aFpsPhases.IsEmpty();
	}

	//------------------------------------------------------------------------------------------------
	//! Places the straight and ring sets. \return empty on success, else the failure
	protected string PlaceMarkers()
	{
		BaseWorld world = GetGame().GetWorld();
		if (!world)
			return "no world";

		array<float> heights = {};
		array<string> tokens = {};
		m_sMarkerHeights.Split(",", tokens, true);
		foreach (string token : tokens)
		{
			string trimmed = token.Trim();
			heights.Insert(trimmed.ToFloat());
		}

		if (heights.IsEmpty())
			heights.Insert(1);

		Resource prefab;
		if (m_eMarkerKind == DC_EProbeMarkerKind.PREFAB)
		{
			prefab = Resource.Load(m_sMarkerPrefab);
			if (!prefab || !prefab.IsValid())
				return "marker prefab " + m_sMarkerPrefab + " cannot be loaded";
		}

		int pair = 0;
		float along = m_fStraightFromM;
		while (along <= m_fStraightToM + 0.001 && m_fStraightStepM > 0)
		{
			float height = heights[pair % heights.Count()];
			string failure = PlaceMarker(world, prefab, "straight", m_Area.PointAt(along, -m_fMarkerLateralM), height);
			if (failure.IsEmpty())
				failure = PlaceMarker(world, prefab, "straight", m_Area.PointAt(along, m_fMarkerLateralM), height);

			if (!failure.IsEmpty())
				return failure;

			pair++;
			along = m_fStraightFromM + pair * m_fStraightStepM;
		}

		for (int i = 0; i < m_iRingCount; i++)
		{
			float angle = Math.PI2 * i / m_iRingCount;
			vector point = m_Area.PointAt(m_fRingAlongM + m_fRingRadiusM * Math.Cos(angle), m_fRingRadiusM * Math.Sin(angle));
			string ringFailure = PlaceMarker(world, prefab, "ring", point, m_fRingHeightM);
			if (!ringFailure.IsEmpty())
				return ringFailure;
		}

		if (m_aMarkers.IsEmpty())
			return "no marker placed (check the straight and ring attributes)";

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! One marker heightM above the ground at pointXZ. \return empty on success, else the failure
	protected string PlaceMarker(notnull BaseWorld world, Resource prefab, string markerSet, vector pointXZ, float heightM)
	{
		DC_ProbeMarker marker = new DC_ProbeMarker();
		marker.m_iId = m_aMarkers.Count();
		marker.m_sSet = markerSet;
		float groundY = world.GetSurfaceY(pointXZ[0], pointXZ[2]);

		if (m_eMarkerKind == DC_EProbeMarkerKind.PREFAB)
		{
			// Prefabs stand on the ground; the logged position is the centre of their local bounds
			EntitySpawnParams params = new EntitySpawnParams();
			params.TransformMode = ETransformMode.WORLD;
			params.Transform[0] = Vector(1, 0, 0);
			params.Transform[1] = Vector(0, 1, 0);
			params.Transform[2] = Vector(0, 0, 1);
			params.Transform[3] = Vector(pointXZ[0], groundY, pointXZ[2]);
			IEntity entity = GetGame().SpawnEntityPrefab(prefab, world, params);
			if (!entity)
				return "SpawnEntityPrefab(" + m_sMarkerPrefab + ") returned null";

			vector mins;
			vector maxs;
			entity.GetBounds(mins, maxs);
			marker.m_Entity = entity;
			marker.m_vSpawnOrigin = entity.GetOrigin();
			marker.m_vCentre = marker.m_vSpawnOrigin + (mins + maxs) * 0.5;
			marker.m_fRadiusM = 0.5 * Math.Max(maxs[0] - mins[0], maxs[2] - mins[2]);
		}
		else
		{
			marker.m_vCentre = Vector(pointXZ[0], groundY + heightM, pointXZ[2]);
			marker.m_vSpawnOrigin = marker.m_vCentre;
			marker.m_fRadiusM = m_fMarkerRadiusM;
			marker.m_Shape = Shape.CreateSphere(GetMarkerColor(), ShapeFlags.NOOUTLINE, marker.m_vCentre, m_fMarkerRadiusM);
			if (!marker.m_Shape)
				return "Shape.CreateSphere returned null";
		}

		m_aMarkers.Insert(marker);
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! \return m_vMarkerColor as an opaque 0xAARRGGBB int
	protected int GetMarkerColor()
	{
		return ARGB(255, ColorByte(m_vMarkerColor[0]), ColorByte(m_vMarkerColor[1]), ColorByte(m_vMarkerColor[2]));
	}

	//------------------------------------------------------------------------------------------------
	//! \return "RRGGBB" of m_vMarkerColor (hex, upper case)
	protected string GetMarkerColorHex()
	{
		int red = ColorByte(m_vMarkerColor[0]);
		int green = ColorByte(m_vMarkerColor[1]);
		int blue = ColorByte(m_vMarkerColor[2]);
		return red.ToString(2, true) + green.ToString(2, true) + blue.ToString(2, true);
	}

	//------------------------------------------------------------------------------------------------
	protected static int ColorByte(float channel)
	{
		int value = Math.Round(Math.Clamp(channel, 0, 1) * 255);
		return value;
	}

	//------------------------------------------------------------------------------------------------
	//! Opens e06a_frames, e06a_phases, e06a_markers_end and e06a_proj. \return false on any failure
	protected bool OpenFiles(notnull DC_ProbeContext ctx)
	{
		array<string> frameColumns = {};
		DC_ProbeImages.AddFrameColumns(frameColumns);
		frameColumns.Insert("fps_setting");
		m_FramesCsv = ctx.OpenCsv("e06a_frames", frameColumns);

		array<string> phaseColumns = {"phase_idx", "fps_setting", "segment", "start_world_ms", "end_world_ms", "start_render_frame", "end_render_frame", "result"};
		m_PhasesCsv = ctx.OpenCsv("e06a_phases", phaseColumns);

		array<string> endColumns = {"marker_id", "phase_idx", "moved", "moved_m"};
		m_MarkersEndCsv = ctx.OpenCsv("e06a_markers_end", endColumns);

		array<string> projColumns = {"render_frame", "marker_id", "screen_x", "screen_y", "depth"};
		m_ProjCsv = ctx.OpenCsv("e06a_proj", projColumns);

		if (!m_FramesCsv || !m_PhasesCsv || !m_MarkersEndCsv || !m_ProjCsv)
			return false;

		return true;
	}

	//------------------------------------------------------------------------------------------------
	protected DC_ProbeCsv OpenRequests(notnull DC_ProbeContext ctx)
	{
		array<string> columns = {};
		DC_ImageSlotScheduler.AddRequestColumns(columns);
		columns.Insert("fps_setting");
		columns.Insert("segment");
		return ctx.OpenCsv("e06a_requests", columns);
	}

	//------------------------------------------------------------------------------------------------
	//! e06a_markers.csv: one row per marker, written once at the start
	protected bool WriteMarkers(notnull DC_ProbeContext ctx)
	{
		array<string> columns = {"marker_id", "set", "x", "y", "z", "radius_m", "color_hex", "kind"};
		DC_ProbeCsv csv = ctx.OpenCsv("e06a_markers", columns);
		if (!csv)
			return false;

		string colorHex = GetMarkerColorHex();
		string kind = "shape";
		if (m_eMarkerKind == DC_EProbeMarkerKind.PREFAB)
			kind = "prefab";

		foreach (DC_ProbeMarker marker : m_aMarkers)
		{
			m_Row.Clear();
			m_Row.AddInt(marker.m_iId);
			m_Row.AddString(marker.m_sSet);
			m_Row.AddVector(marker.m_vCentre, 3);
			m_Row.AddFloat(marker.m_fRadiusM, 3);
			m_Row.AddString(colorHex);
			m_Row.AddString(kind);
			if (!csv.WriteRow(m_Row))
				return false;
		}

		csv.Close();
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! e06a_markers_end.csv rows at the end of an FPS phase; a moved prefab marker stays flagged
	protected void CheckMarkers(int phaseIdx)
	{
		foreach (DC_ProbeMarker marker : m_aMarkers)
		{
			float movedM = 0;
			if (marker.m_Entity)
				movedM = vector.Distance(marker.m_Entity.GetOrigin(), marker.m_vSpawnOrigin);

			if (movedM > MOVED_M && !marker.m_bMoved)
			{
				marker.m_bMoved = true;
				if (m_Ctx)
					m_Ctx.Warn("marker " + marker.m_iId.ToString() + " moved " + movedM.ToString(-1, 2) + " m (excluded by the analysis)");
			}

			if (!m_MarkersEndCsv)
				continue;

			m_Row.Clear();
			m_Row.AddInt(marker.m_iId);
			m_Row.AddInt(phaseIdx);
			m_Row.AddBool(marker.m_bMoved);
			m_Row.AddFloat(movedM, 3);
			m_MarkersEndCsv.WriteRow(m_Row);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! e06a_proj.csv: the engine's projection of up to m_iProjMarkersPerFrame markers in front of the camera
	protected void WriteProjections(int renderFrame)
	{
		if (m_iProjMarkersPerFrame <= 0 || !m_ProjCsv)
			return;

		ArmaReforgerScripted game = GetGame();
		WorkspaceWidget workspace = game.GetWorkspace();
		BaseWorld world = game.GetWorld();
		if (!workspace || !world)
			return;

		vector camera[4];
		m_Rig.GetCameraTransform(camera);
		int written = 0;
		foreach (DC_ProbeMarker marker : m_aMarkers)
		{
			if (written >= m_iProjMarkersPerFrame)
				return;

			vector offset = marker.m_vCentre - camera[3];
			float depth = vector.Dot(offset, camera[2]);
			if (depth <= m_fNearPlaneM || offset.Length() > m_fProjMaxDistM)
				continue;

			vector screen = workspace.ProjWorldToScreenNative(marker.m_vCentre, world);
			m_Row.Clear();
			m_Row.AddInt(renderFrame);
			m_Row.AddInt(marker.m_iId);
			m_Row.AddFloat(screen[0], 3);
			m_Row.AddFloat(screen[1], 3);
			m_Row.AddFloat(screen[2], 3);
			m_ProjCsv.WriteRow(m_Row);
			written++;
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void AppendMarkerLines(notnull array<string> lines)
	{
		int count = m_aMarkers.Count();
		lines.Insert("markers.kind=" + typename.EnumToString(DC_EProbeMarkerKind, m_eMarkerKind));
		lines.Insert("markers.count=" + count.ToString());
		lines.Insert("markers.color_hex=" + GetMarkerColorHex());
		lines.Insert("markers.radius_m=" + m_fMarkerRadiusM.ToString(-1, 3));
		lines.Insert("fps.phases=" + m_sFpsPhases);
		lines.Insert("fps.max_fps_saved=" + BoolText(m_bMaxFpsSaved));
		lines.Insert("fps.max_fps_original=" + m_iMaxFpsOriginal.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! \return the engine video settings module, or null
	protected BaseContainer GetVideoSettings()
	{
		ArmaReforgerScripted game = GetGame();
		if (!game)
			return null;

		UserSettings settings = game.GetEngineUserSettings();
		if (!settings)
			return null;

		return settings.GetModule(VIDEO_MODULE);
	}

	//------------------------------------------------------------------------------------------------
	protected void SaveMaxFps()
	{
		m_bMaxFpsSaved = false;
		m_bMaxFpsChanged = false;
		BaseContainer video = GetVideoSettings();
		if (!video)
		{
			m_Ctx.Warn(VIDEO_MODULE + " not found: MaxFps phases cannot be set (the analysis will show the FPS reached)");
			return;
		}

		int value;
		if (!video.Get(MAX_FPS_KEY, value))
		{
			m_Ctx.Warn(VIDEO_MODULE + "." + MAX_FPS_KEY + " cannot be read: MaxFps phases cannot be set");
			return;
		}

		m_iMaxFpsOriginal = value;
		m_bMaxFpsSaved = true;
	}

	//------------------------------------------------------------------------------------------------
	//! Sets MaxFps (BaseContainer.Set + UserSettingsChanged; never SaveUserSettings). Only after a successful save.
	protected void ApplyMaxFps(int fps)
	{
		if (!m_bMaxFpsSaved)
			return;

		BaseContainer video = GetVideoSettings();
		if (!video || !video.Set(MAX_FPS_KEY, fps))
		{
			if (m_Ctx)
				m_Ctx.Warn("MaxFps " + fps.ToString() + " could not be set");

			return;
		}

		GetGame().UserSettingsChanged();
		m_bMaxFpsChanged = true;
		if (m_Ctx)
			m_Ctx.Log("MaxFps set to " + fps.ToString() + " for FPS phase " + m_iFpsPhase.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! Gives MaxFps its saved value back. Idempotent.
	protected void RestoreMaxFps()
	{
		if (!m_bMaxFpsSaved || !m_bMaxFpsChanged)
			return;

		m_bMaxFpsChanged = false;
		BaseContainer video = GetVideoSettings();
		if (!video || !video.Set(MAX_FPS_KEY, m_iMaxFpsOriginal))
		{
			if (m_Ctx)
				m_Ctx.Warn("MaxFps could not be restored to " + m_iMaxFpsOriginal.ToString());

			return;
		}

		GetGame().UserSettingsChanged();
	}

	//------------------------------------------------------------------------------------------------
	//! Removes markers and bands, gives the camera back and drops every reference. Not for world teardown.
	protected void ReleaseAll()
	{
		foreach (DC_ProbeMarker marker : m_aMarkers)
		{
			if (marker.m_Entity)
				SCR_EntityHelper.DeleteEntityAndChildren(marker.m_Entity);

			marker.m_Entity = null;
			marker.m_Shape = null;
		}

		m_aMarkers.Clear();

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
		m_PhasesCsv = null;
		m_MarkersEndCsv = null;
		m_ProjCsv = null;
		m_Ego = null;
		m_Runner = null;
		m_Ctx = null;
	}

	//------------------------------------------------------------------------------------------------
	protected static string BoolText(bool value)
	{
		if (value)
			return "1";

		return "0";
	}
}
