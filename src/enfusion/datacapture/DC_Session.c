// ============================================================================
// DC_Session - guarded dc-2.0 capture session: lifecycle, provenance and heartbeat (DD-4-min)
// ============================================================================
//
// One component on the capture game mode (GameMode_DataCapture in worlds/DataCapture/DC_Arland). It owns the
// session lifecycle of docs/capture_contract/dc-2.0.md section 5.3 and drives the capture modules: the other
// DC_SessionModule components on the same entity (ego, streams, hf, images). A module never sets its own event mask:
// the session receives every engine event and dispatches it, in the modules' GetDispatchOrder order.
//
// LIFECYCLE (authority only, play mode only; nothing at all in edit mode)
//   1. Guards: edit mode, play mode, server; DC_LaunchGuard.Evaluate (DD-4-probe, reused unchanged). A refusal logs
//      one ERROR line and writes nothing, not even a directory.
//   2. Trial: -dcTrial <id>, else the one-line pointer $profile:dc_trials/ACTIVE; the conf
//      $profile:dc_trials/<id>.json is read with JsonLoadContext (the loader DC_LaunchGuard already uses on 1.8).
//   3. Build and map: GetBuildVersion() must equal the launch record's engine_version_expected once a record carries
//      one (MIGRATION_1_9 8.3); the L8 fingerprint (world size from GetBoundBox, latitude from the weather manager)
//      must match the map card of m_sMapId. Everon is never captured here (sealed test map).
//   4. Readiness: the event mask is set, every module prepares (the ego is spawned and settles, rig R1 attaches, the
//      fiducial bands appear) and is polled once per frame. Meanwhile the physics-step source is counted as in
//      DC_ProbeRunner: "simulate" if any EOnSimulate arrived, else "fixedframe" (WARNING), else none (refusal). A
//      readiness timeout or a module refusal is an ERROR and creates no directory.
//   5. Open: session id <map>_<trial>_<UTC>_<exe>_<inst>_<rand4> (UTC wall clock, rand4 seeded with GetTickCount,
//      never world time); an existing directory is refused (ERROR, nothing written). Then session.json (written once),
//      heartbeat.csv (header), the modules open their streams, events session_start.
//   6. Running: physics_step is -1 at open and +1 at the top of every EOnSimulate (or EOnFixedFrame in the fallback);
//      OnStepPre, OnStepPost (rows), OnStepDecide (episodes, HL decisions) per step; OnFrameStart and OnFrame per
//      render frame. world_time_ms is WorldTimestamp.DiffMilliseconds against the open. warmup_end and the planned
//      end are checked on every step; heartbeat.csv gets a row every m_fHeartbeatPeriodS of world time, and
//      $profile:DC_STOP is checked at each heartbeat.
//   7. End: duration, dc_stop: the modules stop (no new episode or image request) and drain (hf and ctl rows go on
//      until no image request is in flight or the in-flight timeout passed). Play stop and every failure seal at
//      once. Sealing commits every registered stream and writes session_end.json last; status ok only when every
//      commit succeeded and the reason is not an error. Shutdown is idempotent: the event mask is cleared and the
//      only queued call (FallbackStart) is removed, so ScriptCallQueue.Dump() shows no DC_* entry afterwards.
//
// WRITES (all under $profile:DC_Sessions/<session_id>/, never anywhere else; dc-2.0 section 4)
//   session.json, heartbeat.csv and session_end.json go through DC_ProbeCsv.WriteLines (DD-4-probe): every line is
//   one FileHandle.Write whose return must equal the byte count (TBD-3), files are never rewritten in place. The
//   stream chunks belong to the modules' DC_SessionStream writers, which the session owns (RegisterStream) so that it
//   can still commit them on world teardown without calling another component.
//
// TBD PARAMETERS (dc-2.0 section 1; [Attribute]s with the draft default, recorded in session.json): TBD-1 k rule
//   (computed from GetFixedTimeSlice and the decision rate), TBD-2 image rate, in-flight timeout and lead, TBD-3
//   writer semantics (keep logs open, re-check chunk lengths), TBD-6 Workbench escape (off on DC_Arland; the probe
//   layer alone has it on), TBD-8 render-settings modules. Facts the engine reveals at run time (step source, the
//   screenshot name, steps without a world-time advance) are written to the extra "observed" objects of session.json
//   and session_end.json.
//
// FAULT CASES (m_sFaultCase, empty in the layer; for the acceptance runbook only, every one ends in an ERROR or a
//   failed session): readiness_timeout, session_dir_exists, zero_ego, chunk_dir_blocked.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! Ordered JSON text, one member per line, two spaces per level. Strings are escaped; floats are fixed point.
class DC_JsonWriter
{
	//! Both braces in one literal: a lone brace inside a string would unbalance the brace count of the static API
	//! indexer (enfusion/tools/index_api.py keeps string contents)
	static const string BRACES = "{}";

	protected ref array<string> m_aLines = {};
	protected ref array<bool> m_aHasMember = {};	//!< per open container: a member was written already

	//------------------------------------------------------------------------------------------------
	//! Opens an object; key is empty for the root object and for array elements
	void BeginObject(string key)
	{
		AddLine(KeyText(key) + BRACES.Substring(0, 1));
		m_aHasMember.Insert(false);
	}

	//------------------------------------------------------------------------------------------------
	void EndObject()
	{
		CloseContainer(BRACES.Substring(1, 1));
	}

	//------------------------------------------------------------------------------------------------
	void AddString(string key, string value)
	{
		AddLine(KeyText(key) + "\"" + Escape(value) + "\"");
	}

	//------------------------------------------------------------------------------------------------
	//! A string, or null when value is empty
	void AddStringOrNull(string key, string value)
	{
		if (value.IsEmpty())
		{
			AddNull(key);
			return;
		}

		AddString(key, value);
	}

	//------------------------------------------------------------------------------------------------
	void AddInt(string key, int value)
	{
		AddLine(KeyText(key) + value.ToString());
	}

	//------------------------------------------------------------------------------------------------
	//! value.ToString(-1, decimals): fixed point, never an exponent
	void AddFloat(string key, float value, int decimals)
	{
		AddLine(KeyText(key) + value.ToString(-1, decimals));
	}

	//------------------------------------------------------------------------------------------------
	void AddBool(string key, bool value)
	{
		if (value)
			AddLine(KeyText(key) + "true");
		else
			AddLine(KeyText(key) + "false");
	}

	//------------------------------------------------------------------------------------------------
	void AddNull(string key)
	{
		AddLine(KeyText(key) + "null");
	}

	//------------------------------------------------------------------------------------------------
	//! [a, b, ...] on one line
	void AddIntArray(string key, notnull array<int> values)
	{
		string text = "[";
		for (int i = 0, count = values.Count(); i < count; i++)
		{
			if (i > 0)
				text += ", ";

			int value = values[i];
			text += value.ToString();
		}

		AddLine(KeyText(key) + text + "]");
	}

	//------------------------------------------------------------------------------------------------
	//! The lines written so far (complete JSON once every object is closed)
	array<string> GetLines()
	{
		return m_aLines;
	}

	//------------------------------------------------------------------------------------------------
	//! JSON string escape: backslash, quote, CR, LF, TAB
	static string Escape(string value)
	{
		string text = value;
		text.Replace("\\", "\\\\");
		text.Replace("\"", "\\\"");
		text.Replace("\r", "\\r");
		text.Replace("\n", "\\n");
		text.Replace("\t", "\\t");
		return text;
	}

	//------------------------------------------------------------------------------------------------
	protected string KeyText(string key)
	{
		if (key.IsEmpty())
			return string.Empty;

		return "\"" + Escape(key) + "\": ";
	}

	//------------------------------------------------------------------------------------------------
	//! Appends one member line at the current depth, with the comma on the previous member
	protected void AddLine(string text)
	{
		int depth = m_aHasMember.Count();
		if (depth > 0)
		{
			if (m_aHasMember[depth - 1])
				m_aLines[m_aLines.Count() - 1] = m_aLines[m_aLines.Count() - 1] + ",";

			m_aHasMember[depth - 1] = true;
		}

		m_aLines.Insert(Indent(depth) + text);
	}

	//------------------------------------------------------------------------------------------------
	protected void CloseContainer(string closing)
	{
		int depth = m_aHasMember.Count();
		if (depth == 0)
			return;

		m_aHasMember.Remove(depth - 1);
		m_aLines.Insert(Indent(depth - 1) + closing);
	}

	//------------------------------------------------------------------------------------------------
	protected static string Indent(int depth)
	{
		string text;
		for (int i = 0; i < depth; i++)
		{
			text += "  ";
		}

		return text;
	}
}

//------------------------------------------------------------------------------------------------
//! A stream the session commits, counts and closes. DC_ChunkWriter implements it; a module creates its writers and
//! hands them to DC_Session.RegisterStream, which keeps the only strong reference.
class DC_SessionStream
{
	//------------------------------------------------------------------------------------------------
	//! Stream name of the contract (ctl, telemetry, hf, img, events, episodes)
	string GetName()
	{
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! Buffers one row with its key; false on a failed write (HasFailed)
	bool AddRow(notnull DC_ProbeRow row, int key, int worldMs)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Commits the buffered chunk once it is clocks.chunk_max_s old; false on a failed write
	bool FlushIfDue(int worldMs)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Commits what is buffered and closes the stream. Idempotent; false when any write of the stream failed
	bool Close()
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Drops what is buffered without writing it (never acknowledged) and closes. Idempotent
	void Abandon()
	{
	}

	//------------------------------------------------------------------------------------------------
	bool HasFailed()
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	string GetFailure()
	{
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! Rows acknowledged by a commit line
	int GetCommittedRows()
	{
		return 0;
	}

	//------------------------------------------------------------------------------------------------
	int GetCommittedChunks()
	{
		return 0;
	}

	//------------------------------------------------------------------------------------------------
	//! Key of the last acknowledged row, -1 without one
	int GetLastCommittedKey()
	{
		return -1;
	}
}

//------------------------------------------------------------------------------------------------
//! Base of every capture module on the capture game mode. DC_Session discovers them with
//! FindComponents(DC_SessionModule, ...) and calls the hooks below; a module never sets its own event mask.
class DC_SessionModuleClass : ScriptComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_SessionModule : ScriptComponent
{
	//------------------------------------------------------------------------------------------------
	//! Modules are called in ascending order (ego 10, streams 20, hf 30, images 40)
	int GetDispatchOrder()
	{
		return 100;
	}

	//------------------------------------------------------------------------------------------------
	//! Readiness starts (no directory yet). false = refused (the module logs why through the session)
	bool OnSessionPrepare(notnull DC_Session session)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Polled once per frame until every module is ready. waitingFor names what is missing (for the timeout ERROR)
	bool IsSessionReady(out string waitingFor)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Writes this module's keys into the session.json object named section ("rig", "control")
	void WriteSessionJson(string section, notnull DC_JsonWriter json)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! This module's part of rig.rig_id (empty = none)
	string GetRigIdPart()
	{
		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! The directory and session.json exist: create the stream directories and writers (DC_Session.RegisterStream).
	//! false = the session fails (script_error, or write_fail when a write failure was reported)
	bool OnSessionOpen(notnull DC_Session session)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Top of a physics step (EOnSimulate, or EOnFixedFrame in the fallback): the held command is applied here
	void OnStepPre(int physicsStep, float timeSlice)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Same step after the simulation: the rows of the step
	void OnStepPost(int physicsStep, float timeSlice, int worldMs)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! After every module's OnStepPost of the step: episode boundaries and HL decisions (dc-2.0 section 6)
	void OnStepDecide(int physicsStep)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! First call of every render frame, before any row of it (the fiducial is drawn here)
	void OnFrameStart(int renderFrame)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Every render frame after OnFrameStart
	void OnFrame(int renderFrame, float timeSlice, int worldMs)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! The planned end began (duration, dc_stop): no new episode or image request from now on
	void OnSessionStopping(string endReason)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Polled every frame while the session drains; true when nothing of this module is in flight any more
	bool IsDrained()
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Before the streams are committed: the last rows (the episodes row of an episode still open)
	void OnSessionClosing(string endReason)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! After session_end.json, or after a refusal: give the camera back, remove widgets, park the ego. teardown =
	//! world deletion: drop references only and touch no other entity, camera or widget. Idempotent
	void OnSessionRelease(bool teardown)
	{
	}
}

//------------------------------------------------------------------------------------------------
enum DC_ESessionPhase
{
	IDLE,			//!< nothing started (or not the authority)
	PREPARING,		//!< readiness: no directory yet
	CAPTURING,		//!< directory open, rows written
	DRAINING,		//!< planned end: no new image request, rows go on until the modules are drained
	CLOSED			//!< sealed, refused or stopped; nothing is captured any more
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "dc-2.0 capture session: launch guard, trial, provenance, readiness, physics-step and frame dispatch to the capture modules, heartbeat and seal")]
class DC_SessionClass : SCR_BaseGameModeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_Session : SCR_BaseGameModeComponent
{
	// === MAP (declared by the capture world, verified by the L8 fingerprint) ===
	[Attribute("arland", UIWidgets.EditBox, "Canonical map id of this capture world (arland or kolguyev; everon is never captured by this component)", category: "Map")]
	protected string m_sMapId;

	[Attribute("Arland", UIWidgets.EditBox, "Engine terrain name of the root world (session.json map.terrain)", category: "Map")]
	protected string m_sTerrain;

	[Attribute("{A9806AF617972E97}worlds/Arland/Arland.ent", UIWidgets.EditBox, "Root world resource of the terrain (dc-2.0 section 3 table)", category: "Map")]
	protected string m_sRootWorld;

	[Attribute("4096", UIWidgets.Slider, "L8 fingerprint: expected world size in x [m] (program/maps/map_cards_v1.json)", "100 50000 1", category: "Map")]
	protected float m_fMapSizeM;

	[Attribute("256", UIWidgets.Slider, "L8 fingerprint: allowed difference of the GetBoundBox x extent from the expected size [m]", "0 10000 1", category: "Map")]
	protected float m_fMapSizeTolM;

	[Attribute("47.411", UIWidgets.Slider, "L8 fingerprint: expected latitude of the weather manager [deg] (map card default_latitude)", "-90 90 0.001", category: "Map")]
	protected float m_fLatitudeDeg;

	[Attribute("0.05", UIWidgets.Slider, "L8 fingerprint: allowed latitude difference [deg]", "0 10 0.001", category: "Map")]
	protected float m_fLatitudeTolDeg;

	// === LAUNCH GUARD (DC_LaunchGuard of DD-4-probe) ===
	[Attribute("E:/DriveDiT", UIWidgets.EditBox, "Launch guard: -profile and $profile must lie inside this folder (case-insensitive, either slash). Empty refuses: the guard has no off switch", category: "Launch")]
	protected string m_sRequiredProfileRoot;

	[Attribute("0", UIWidgets.CheckBox, "Launch guard: Workbench-only escape for a Play that cannot see the command line (TBD-6 / E-11). Off on DC_Arland (the probe layer alone has it on); escape sessions are dev data, marked launch_guard.launch_mode wb_escape", category: "Launch")]
	protected bool m_bAllowWorkbenchWithoutLaunchId;

	[Attribute("1", UIWidgets.CheckBox, "Launch guard escape only: the newest -wb- record must be this Workbench start's own record (DC_LaunchGuard G15)", category: "Launch")]
	protected bool m_bEscapeBindToProcessStart;

	// === CLOCKS (TBD-1, TBD-2) ===
	[Attribute("10", UIWidgets.Slider, "Image slot rate [Hz] (TBD-2: 10, or the 5 Hz fallback). The HL decision rate equals it; physics_steps_per_decision and telemetry_every_k follow the TBD-1 rule", "1 30 1", category: "Clocks")]
	protected float m_fImageHz;

	[Attribute("2000", UIWidgets.Slider, "A screenshot request not seen on disk after this long [ms of real time] stops blocking the next slot (TBD-2)", "100 60000 100", category: "Clocks")]
	protected int m_iImgInflightTimeoutMs;

	[Attribute("0", UIWidgets.Slider, "Image grid lead [ms]: grid point g is due this long before decision step g (TBD-2, from the E-05/E-06 offsets)", "0 1000 1", category: "Clocks")]
	protected int m_iImgLeadMs;

	[Attribute("10", UIWidgets.Slider, "A chunk is committed at the latest after this much world time [s] (clocks.chunk_max_s)", "1 60 1", category: "Clocks")]
	protected float m_fChunkMaxS;

	// === LIFECYCLE ===
	[Attribute("30", UIWidgets.Slider, "Readiness must complete within this much world time [s], else ERROR and no directory", "5 300 1", category: "Lifecycle")]
	protected float m_fReadinessTimeoutS;

	[Attribute("5", UIWidgets.Slider, "Heartbeat period [s of world time] (heartbeat.csv, DC_STOP check)", "1 60 1", category: "Lifecycle")]
	protected float m_fHeartbeatPeriodS;

	[Attribute("10000", UIWidgets.Slider, "Start this long after OnPostInit if OnGameModeStart never arrived [ms], 0 = no fallback", "0 60000 500", category: "Lifecycle")]
	protected int m_iFallbackStartMs;

	[Attribute("1", UIWidgets.CheckBox, "After a DC_STOP seal, request the game to close (never in Workbench)", category: "Lifecycle")]
	protected bool m_bCloseOnDcStop;

	// === WRITER (TBD-3) ===
	[Attribute("0", UIWidgets.CheckBox, "TBD-3 fallback: keep commit.csv and heartbeat.csv open for the session instead of one open-APPEND-write-close cycle per line (the e19kill e19_commit.csv pattern)", category: "Writer")]
	protected bool m_bKeepLogsOpen;

	[Attribute("0", UIWidgets.CheckBox, "TBD-3 fallback: after a chunk is closed, read its length back through a READ handle before the commit line (when E-19 shows that Write does not report short writes)", category: "Writer")]
	protected bool m_bVerifyChunkLength;

	// === PROVENANCE (TBD-8) ===
	[Attribute("VideoUserSettings,DisplayUserSettings", UIWidgets.EditBox, "Engine user-settings modules dumped into rig.render_settings and hashed (TBD-8, as E-20)", category: "Provenance")]
	protected string m_sRenderSettingsModules;

	// === FAULT CASES (acceptance runbook only) ===
	[Attribute("", UIWidgets.EditBox, "Fault case for the acceptance runbook, empty in the layer: readiness_timeout, session_dir_exists, zero_ego or chunk_dir_blocked. Every case ends in an ERROR or a failed session", category: "Faults")]
	protected string m_sFaultCase;

	// === CONSTANTS ===
	static const string CONTRACT_VERSION = "dc-2.0";
	static const string FEATURES_VERSION = "features_v1";
	static const string NAV_VERSION = "nav_v1";
	static const string SESSION_ROOT = "$profile:DC_Sessions";
	static const string TRIAL_DIR = "$profile:dc_trials";
	static const string TRIAL_ACTIVE = "$profile:dc_trials/ACTIVE";
	static const string STOP_FILE = "$profile:DC_STOP";
	static const string SESSION_JSON = "session.json";
	static const string SESSION_END_JSON = "session_end.json";
	static const string HEARTBEAT_CSV = "heartbeat.csv";
	static const string HEARTBEAT_HEADER = "unix_s,world_ms,physics_step,render_frame,fps,mem_kb,img_ok,img_fail";
	static const string CLI_TRIAL = "dcTrial";
	static const string CLI_INSTANCE = "dcInstance";
	static const string LOG = "[DC_Session] ";

	//! dc-2.0 section 6 / schema: sessions are at most 30 min (the fiducial code is unique for 73 min at 60 FPS)
	static const float MAX_SESSION_S = 1800;
	//! TBD-1 rule: the telemetry period is at most 50 ms (+1 us)
	static const float TELEMETRY_MAX_PERIOD_S = 0.050001;

	static const string STEP_SIMULATE = "simulate";
	static const string STEP_FIXEDFRAME = "fixedframe";
	static const string STEP_NONE = "none";

	static const string END_DURATION = "duration";
	static const string END_STOP = "stop";
	static const string END_DC_STOP = "dc_stop";
	static const string END_WRITE_FAIL = "script_error:write_fail";
	static const string END_SCRIPT_ERROR = "script_error:other";
	static const string END_ZERO_EGO = "zero_ego";
	static const string END_EGO_LOST = "ego_lost";
	static const string END_CAMERA_LOST = "camera_lost";

	static const string FAULT_READINESS_TIMEOUT = "readiness_timeout";
	static const string FAULT_SESSION_DIR_EXISTS = "session_dir_exists";
	static const string FAULT_ZERO_EGO = "zero_ego";
	static const string FAULT_CHUNK_DIR_BLOCKED = "chunk_dir_blocked";

	//! events.kind values of dc-2.0 section 9.5
	protected static ref array<string> s_aEventKinds = {"session_start", "warmup_end", "episode_start", "settle_end", "episode_end", "perturb_start", "perturb_end", "nav_command", "style_set", "teleport", "respawn", "env_change", "pause", "resume", "stuck", "collision", "camera_reset", "img_timeout", "dc_stop", "script_error", "veh_cfg_override"};

	// === STATE ===
	protected bool m_bStarted;					//!< StartRun ran (any machine), never cleared
	protected bool m_bMaskSet;
	protected float m_fFallbackArmedMs;
	protected DC_ESessionPhase m_ePhase = DC_ESessionPhase.IDLE;
	protected ref DC_LaunchInfo m_LaunchInfo;
	protected ref array<DC_SessionModule> m_aModules = {};		//!< sorted by GetDispatchOrder
	protected ref array<ref DC_SessionStream> m_aStreams = {};	//!< the only strong references to the writers

	// trial
	protected string m_sTrialId;
	protected string m_sTrialSource;
	protected string m_sCliTrial;
	protected string m_sTrialRoute;
	protected float m_fWarmupS;
	protected float m_fDurationS;
	protected int m_iTrialSeed;
	protected bool m_bTrialImages = true;

	// build and map
	protected string m_sBuildVersion;
	protected string m_sEngineVersionExpected;
	protected float m_fMeasuredSizeM;
	protected float m_fMeasuredLatitudeDeg;
	protected float m_fFixedTimeSliceS;
	protected int m_iStepsPerDecision = 1;
	protected int m_iTelemetryEveryK = 1;

	// readiness
	protected float m_fPrepareStartMs;
	protected string m_sReadinessRefusal;
	protected string m_sStepSource;
	protected int m_iSimulateCount;
	protected int m_iPostSimulateCount;
	protected int m_iFixedFrameCount;
	protected int m_iPostFixedFrameCount;
	protected int m_iFrameCount;

	// session
	protected string m_sSessionId;
	protected string m_sSessionDir;
	protected string m_sCreatedUtc;
	protected string m_sExeKind;
	protected string m_sInstance;
	protected WorldTimestamp m_StartTimestamp;
	protected bool m_bHasStartTimestamp;
	protected int m_iStartUnix;
	protected int m_iPhysicsStep = -1;			//!< -1 at open, +1 at the top of every dispatched step
	protected int m_iLastPostStep = -1;			//!< last step whose OnStepPost was dispatched
	protected int m_iStepZeroWorldMs = -1;		//!< world_time_ms of physics step 0 (image grid anchor)
	protected int m_iLastStepWorldMs = -1;
	protected int m_iEpisodeId = -1;
	protected bool m_bWarmupOver;
	protected bool m_bStopping;
	protected string m_sEndReason;
	protected string m_sFailureReason;			//!< first failure: an error end reason
	protected string m_sFailureDetail;
	protected bool m_bWriteFailed;
	protected int m_iDrainStartTickMs;
	protected int m_iDrainStartWorldMs;
	protected int m_iNextHeartbeatMs;
	protected int m_iHeartbeats;
	protected ref FileHandle m_HeartbeatFile;	//!< only with m_bKeepLogsOpen
	protected int m_iEventSeq;
	protected ref DC_ProbeRow m_EventRow = new DC_ProbeRow();
	protected int m_iPostWithoutPre;
	protected int m_iPreWithoutPost;
	protected int m_iStepsWithoutWorldAdvance;

	// ego (set by the ego module)
	protected IEntity m_Ego;
	protected int m_iEgoSlotId = -1;
	protected string m_sEgoEntityId;
	protected string m_sEgoPrefab;

	// image counts (set by the image module; session_end.json images and heartbeat img_ok / img_fail)
	protected int m_iImagesRequested;
	protected int m_iImagesIssued;
	protected int m_iImagesSkippedBusy;
	protected int m_iImagesMakeFailed;
	protected int m_iImagesTimedOut;
	protected int m_iImagesPending;
	protected int m_iImagesOk;
	protected int m_iImagesFailed;

	// facts seen at run time (session_end.json observed)
	protected ref array<string> m_aObservedKeys = {};
	protected ref array<string> m_aObservedValues = {};

	//------------------------------------------------------------------------------------------------
	override void OnPostInit(IEntity owner)
	{
		super.OnPostInit(owner);

		// OnPostInit also runs when the World Editor loads the layer: nothing at all in edit mode (CR-012)
		if (SCR_Global.IsEditMode(owner) || SCR_Global.IsEditMode())
			return;

		if (m_iFallbackStartMs > 0)
		{
			BaseWorld world = owner.GetWorld();
			if (world)
				m_fFallbackArmedMs = world.GetWorldTime();

			GetGame().GetCallqueue().CallLater(FallbackStart, m_iFallbackStartMs, false);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Called from SCR_BaseGameMode.OnGameStateChanged() when the state becomes GAME (on every machine)
	override void OnGameModeStart()
	{
		super.OnGameModeStart();
		StartRun("OnGameModeStart");
	}

	//------------------------------------------------------------------------------------------------
	//! As DC_ProbeRunner.FallbackStart: only once m_iFallbackStartMs of world time passed since OnPostInit
	protected void FallbackStart()
	{
		if (m_bStarted)
			return;

		BaseWorld world = GetGame().GetWorld();
		if (world)
		{
			int remainingMs = m_iFallbackStartMs - (world.GetWorldTime() - m_fFallbackArmedMs);
			if (remainingMs > 0)
			{
				if (remainingMs < 100)
					remainingMs = 100;

				GetGame().GetCallqueue().CallLater(FallbackStart, remainingMs, false);
				return;
			}
		}

		Print(string.Format(LOG + "WARNING: OnGameModeStart not received within %1 ms", m_iFallbackStartMs), LogLevel.WARNING);
		StartRun("FallbackStart");
	}

	//------------------------------------------------------------------------------------------------
	//! Steps 1-4 of the lifecycle. Idempotent: only the first call does anything.
	protected void StartRun(string reason)
	{
		if (m_bStarted)
			return;

		if (SCR_Global.IsEditMode() || !GetGame().InPlayMode())
			return;

		m_bStarted = true;
		GetGame().GetCallqueue().Remove(FallbackStart);

		if (!Replication.IsServer())
		{
			Print(LOG + "not the server, nothing to do on this machine", LogLevel.NORMAL);
			return;
		}

		// The guard never prints, so the E-11 facts come first. The line shows $profile: never paste it into a report
		m_LaunchInfo = DC_LaunchGuard.Evaluate(m_sRequiredProfileRoot, m_bAllowWorkbenchWithoutLaunchId, m_bEscapeBindToProcessStart);
		string profileAbs = m_LaunchInfo.m_sProfileAbs;
		if (profileAbs.IsEmpty())
			profileAbs = "n/a";

		Print(string.Format(LOG + "E-11: -profile visible=%1 -dcLaunchId visible=%2 $profile=%3 exe=%4 uptime_s=%5 (%6)", DC_LaunchGuard.BoolText(m_LaunchInfo.m_bCliProfile), DC_LaunchGuard.BoolText(m_LaunchInfo.m_bCliLaunchId), profileAbs, GetExeKind(), m_LaunchInfo.m_iUptimeS, reason), LogLevel.NORMAL);
		if (!m_LaunchInfo.m_bAllowed)
		{
			Refuse("launch guard refused: " + m_LaunchInfo.m_sReason);
			return;
		}

		string guardLine = string.Format(LOG + "launch guard: mode=%1 id=%2 pin=%3 use_class=%4", m_LaunchInfo.m_sMode, m_LaunchInfo.m_sLaunchId, m_LaunchInfo.m_sPinId, m_LaunchInfo.m_sUseClass);
		if (m_LaunchInfo.m_sMode == DC_LaunchGuard.MODE_ESCAPE)
			Print(guardLine + " (Workbench escape, binding " + m_LaunchInfo.m_sEscapeBinding + ": dev data only)", LogLevel.WARNING);
		else
			Print(guardLine, LogLevel.NORMAL);

		string failure;
		if (!ResolveTrial(failure))
		{
			Refuse("trial: " + failure);
			return;
		}

		if (!CheckBuild(failure))
		{
			Refuse("build: " + failure);
			return;
		}

		if (!CheckFingerprint(failure))
		{
			Refuse("L8 map fingerprint: " + failure);
			return;
		}

		if (!ComputeClocks(failure))
		{
			Refuse("clocks: " + failure);
			return;
		}

		CollectModules();
		if (m_aModules.IsEmpty())
		{
			Refuse("no DC_SessionModule component on " + GetOwner().GetName());
			return;
		}

		BeginReadiness();
	}

	//------------------------------------------------------------------------------------------------
	//! -dcTrial <id>, else the ACTIVE pointer; then $profile:dc_trials/<id>.json (dc-2.0 section 5.2)
	protected bool ResolveTrial(out string failure)
	{
		string cliTrial;
		if (System.GetCLIParam(CLI_TRIAL, cliTrial) && !cliTrial.Trim().IsEmpty())
		{
			m_sTrialId = cliTrial.Trim();
			m_sCliTrial = m_sTrialId;
			m_sTrialSource = "cli";
		}
		else
		{
			if (!FileIO.FileExists(TRIAL_ACTIVE))
			{
				failure = "no -dcTrial and no " + TRIAL_ACTIVE + " pointer";
				return false;
			}

			FileHandle activeFile = FileIO.OpenFile(TRIAL_ACTIVE, FileMode.READ);
			if (!activeFile)
			{
				failure = "cannot open " + TRIAL_ACTIVE;
				return false;
			}

			string line;
			activeFile.ReadLine(line);
			activeFile.Close();
			m_sTrialId = line.Trim();
			m_sTrialSource = "active_pointer";
		}

		if (!IsValidTrialId(m_sTrialId))
		{
			failure = "trial id '" + m_sTrialId + "' is not 1-16 characters of [A-Za-z0-9-]";
			return false;
		}

		string confPath = TRIAL_DIR + "/" + m_sTrialId + ".json";
		if (!FileIO.FileExists(confPath))
		{
			failure = "no trial conf " + confPath;
			return false;
		}

		JsonLoadContext context = new JsonLoadContext(false);
		if (!context.LoadFromFile(confPath))
		{
			failure = "trial conf " + confPath + " does not parse";
			return false;
		}

		string confTrialId;
		if (!context.ReadValue("trial_id", confTrialId) || confTrialId != m_sTrialId)
		{
			failure = "trial conf " + confPath + " names trial '" + confTrialId + "'";
			return false;
		}

		float durationS;
		float warmupS;
		if (!ReadNumber(context, "duration_s", durationS) || !ReadNumber(context, "warmup_s", warmupS))
		{
			failure = "trial conf " + confPath + " misses duration_s or warmup_s";
			return false;
		}

		int seed;
		if (!context.ReadValue("seed", seed))
		{
			failure = "trial conf " + confPath + " misses the integer seed";
			return false;
		}

		m_fDurationS = durationS;
		m_fWarmupS = warmupS;
		m_iTrialSeed = seed;

		string route;
		if (context.DoesKeyExist("route") && context.ReadValue("route", route))
			m_sTrialRoute = route;

		bool images;
		if (context.DoesKeyExist("images") && context.ReadValue("images", images))
			m_bTrialImages = images;

		if (m_fDurationS <= 0 || m_fWarmupS < 0 || m_fWarmupS + m_fDurationS > MAX_SESSION_S)
		{
			failure = string.Format("warmup_s %1 + duration_s %2 must be positive and at most %3 s", m_fWarmupS, m_fDurationS, MAX_SESSION_S);
			return false;
		}

		Print(string.Format(LOG + "trial %1 (%2): warm-up %3 s, duration %4 s, seed %5, route '%6', images %7", m_sTrialId, m_sTrialSource, m_fWarmupS.ToString(-1, 1), m_fDurationS.ToString(-1, 1), m_iTrialSeed, m_sTrialRoute, DC_LaunchGuard.BoolText(m_bTrialImages)), LogLevel.NORMAL);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! A JSON number as float: written as a decimal or as an integer
	protected bool ReadNumber(notnull JsonLoadContext context, string key, out float value)
	{
		if (!context.DoesKeyExist(key))
			return false;

		if (context.ReadValue(key, value))
			return true;

		int integer;
		if (!context.ReadValue(key, integer))
			return false;

		value = integer;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! GetBuildVersion() against the launch record's engine_version_expected, once a record carries one (top level or
	//! pin.engine_version_expected; DD-14a). Without it the check is recorded as not done.
	protected bool CheckBuild(out string failure)
	{
		m_sBuildVersion = GetGame().GetBuildVersion();
		JsonLoadContext context = new JsonLoadContext(false);
		if (!context.LoadFromFile(m_LaunchInfo.m_sRecordPath))
		{
			failure = "launch record " + m_LaunchInfo.m_sRecordPath + " no longer parses";
			return false;
		}

		string expected;
		if (context.DoesKeyExist("engine_version_expected") && context.ReadValue("engine_version_expected", expected))
			m_sEngineVersionExpected = expected;

		if (m_sEngineVersionExpected.IsEmpty() && context.DoesKeyExist("pin") && context.StartObject("pin"))
		{
			if (context.DoesKeyExist("engine_version_expected") && context.ReadValue("engine_version_expected", expected))
				m_sEngineVersionExpected = expected;

			context.EndObject();
		}

		if (m_sEngineVersionExpected.IsEmpty())
		{
			Print(LOG + "build " + m_sBuildVersion + " (pin " + m_LaunchInfo.m_sPinId + " records no engine_version_expected yet: not compared)", LogLevel.NORMAL);
			return true;
		}

		if (m_sBuildVersion != m_sEngineVersionExpected)
		{
			failure = "GetBuildVersion() is " + m_sBuildVersion + ", pin " + m_LaunchInfo.m_sPinId + " expects " + m_sEngineVersionExpected;
			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! L8 (dc-2.0 3.1): map_id is declared by the capture world and checked against the world size (GetBoundBox x
	//! extent) and the weather manager's latitude; never parsed from GetWorldFile() (CR-100)
	protected bool CheckFingerprint(out string failure)
	{
		if (m_sMapId != "arland" && m_sMapId != "kolguyev")
		{
			failure = "map id '" + m_sMapId + "' is not a capture map of this component (arland, kolguyev)";
			return false;
		}

		BaseWorld world = GetGame().GetWorld();
		ChimeraWorld chimeraWorld = ChimeraWorld.CastFrom(world);
		if (!world || !chimeraWorld)
		{
			failure = "no world";
			return false;
		}

		vector boundsMin;
		vector boundsMax;
		world.GetBoundBox(boundsMin, boundsMax);
		m_fMeasuredSizeM = boundsMax[0] - boundsMin[0];

		TimeAndWeatherManagerEntity weather = chimeraWorld.GetTimeAndWeatherManager();
		if (!weather)
		{
			failure = "no TimeAndWeatherManagerEntity: the latitude cannot be checked";
			return false;
		}

		m_fMeasuredLatitudeDeg = weather.GetCurrentLatitude();
		string facts = string.Format("size %1 m (expected %2 +- %3), latitude %4 deg (expected %5 +- %6)", m_fMeasuredSizeM.ToString(-1, 1), m_fMapSizeM.ToString(-1, 1), m_fMapSizeTolM.ToString(-1, 1), m_fMeasuredLatitudeDeg.ToString(-1, 4), m_fLatitudeDeg.ToString(-1, 4), m_fLatitudeTolDeg.ToString(-1, 4));
		if (Math.AbsFloat(m_fMeasuredSizeM - m_fMapSizeM) > m_fMapSizeTolM || Math.AbsFloat(m_fMeasuredLatitudeDeg - m_fLatitudeDeg) > m_fLatitudeTolDeg)
		{
			failure = "world " + GetGame().GetWorldFile() + " is not map " + m_sMapId + ": " + facts;
			return false;
		}

		Print(LOG + "L8 fingerprint ok for " + m_sMapId + ": " + facts, LogLevel.NORMAL);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! TBD-1 in this order: N = floor(1 / decision_hz / slice + 0.5), then k = the largest divisor of N with
	//! k x slice <= 50 ms (+1 us), at least 1. The decision rate is the image rate (TBD-2)
	protected bool ComputeClocks(out string failure)
	{
		m_fFixedTimeSliceS = GetGame().GetWorld().GetFixedTimeSlice();
		if (m_fFixedTimeSliceS <= 0 || m_fImageHz <= 0)
		{
			failure = string.Format("fixed time slice %1 s, image rate %2 Hz", m_fFixedTimeSliceS, m_fImageHz);
			return false;
		}

		m_iStepsPerDecision = Math.Floor(1.0 / m_fImageHz / m_fFixedTimeSliceS + 0.5);
		if (m_iStepsPerDecision < 1)
			m_iStepsPerDecision = 1;

		m_iTelemetryEveryK = 1;
		for (int divisor = 1; divisor <= m_iStepsPerDecision; divisor++)
		{
			if (m_iStepsPerDecision % divisor == 0 && divisor * m_fFixedTimeSliceS <= TELEMETRY_MAX_PERIOD_S)
				m_iTelemetryEveryK = divisor;
		}

		Print(string.Format(LOG + "clocks: slice %1 s, decisions and images at %2 Hz, N=%3 steps per decision, telemetry every k=%4 steps", m_fFixedTimeSliceS.ToString(-1, 6), m_fImageHz.ToString(-1, 1), m_iStepsPerDecision, m_iTelemetryEveryK), LogLevel.NORMAL);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Every DC_SessionModule on the game mode, sorted by GetDispatchOrder (insertion sort, stable)
	protected void CollectModules()
	{
		array<Managed> found = {};
		GetOwner().FindComponents(DC_SessionModule, found);
		foreach (Managed managed : found)
		{
			DC_SessionModule module = DC_SessionModule.Cast(managed);
			if (!module)
				continue;

			int index = m_aModules.Count();
			while (index > 0 && m_aModules[index - 1].GetDispatchOrder() > module.GetDispatchOrder())
			{
				index--;
			}

			m_aModules.InsertAt(module, index);
		}

		string names;
		foreach (DC_SessionModule listed : m_aModules)
		{
			if (!names.IsEmpty())
				names += ", ";

			names += listed.ClassName();
		}

		Print(string.Format(LOG + "%1 capture module(s): %2", m_aModules.Count(), names), LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! Step 4: event mask on, modules prepare; UpdateReadiness polls them once per frame
	protected void BeginReadiness()
	{
		m_ePhase = DC_ESessionPhase.PREPARING;
		m_fPrepareStartMs = GetGame().GetWorld().GetWorldTime();
		SetEventMask(GetOwner(), GetDispatchEventMask());
		m_bMaskSet = true;

		foreach (DC_SessionModule module : m_aModules)
		{
			if (!module.OnSessionPrepare(this))
			{
				string reason = m_sReadinessRefusal;
				if (reason.IsEmpty())
					reason = module.ClassName() + " refused to prepare";

				Refuse("readiness: " + reason);
				return;
			}
		}

		Print(string.Format(LOG + "readiness: waiting up to %1 s for the ego, the camera and the step source", m_fReadinessTimeoutS.ToString(-1, 0)), LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! Once per frame while PREPARING: open the session when every module is ready and a step source exists
	protected void UpdateReadiness()
	{
		if (!m_sReadinessRefusal.IsEmpty())
		{
			Refuse("readiness: " + m_sReadinessRefusal);
			return;
		}

		string waiting;
		foreach (DC_SessionModule module : m_aModules)
		{
			string waitingFor;
			if (module.IsSessionReady(waitingFor))
				continue;

			if (!waiting.IsEmpty())
				waiting += "; ";

			waiting += module.ClassName() + ": " + waitingFor;
		}

		if (!m_sReadinessRefusal.IsEmpty())
		{
			Refuse("readiness: " + m_sReadinessRefusal);
			return;
		}

		float elapsedS = (GetGame().GetWorld().GetWorldTime() - m_fPrepareStartMs) / 1000;
		if (waiting.IsEmpty())
		{
			if (DecideStepSource(elapsedS))
				OpenSession();
			else
				Refuse("readiness: no physics step source (neither EOnSimulate nor EOnFixedFrame arrived on the game mode)");

			return;
		}

		if (elapsedS >= m_fReadinessTimeoutS)
			Refuse(string.Format("readiness timeout after %1 s, waiting for %2", elapsedS.ToString(-1, 1), waiting));
	}

	//------------------------------------------------------------------------------------------------
	//! As DC_ProbeRunner.DecideStepSource: simulate if any EOnSimulate arrived, else fixedframe, else none
	protected bool DecideStepSource(float elapsedS)
	{
		string counts = string.Format("EOnSimulate %1, EOnPostSimulate %2, EOnFixedFrame %3, EOnPostFixedFrame %4, EOnFrame %5 in %6 s", m_iSimulateCount, m_iPostSimulateCount, m_iFixedFrameCount, m_iPostFixedFrameCount, m_iFrameCount, elapsedS.ToString(-1, 2));
		if (m_iSimulateCount > 0)
		{
			m_sStepSource = STEP_SIMULATE;
			Print(LOG + "step source: simulate (" + counts + ")", LogLevel.NORMAL);
			return true;
		}

		if (m_iFixedFrameCount > 0)
		{
			m_sStepSource = STEP_FIXEDFRAME;
			Print(LOG + "WARNING: step source: fixedframe - no EOnSimulate on the game mode entity, physics steps come from EOnFixedFrame (" + counts + ")", LogLevel.WARNING);
			return true;
		}

		m_sStepSource = STEP_NONE;
		Print(LOG + "ERROR: step source: none (" + counts + ")", LogLevel.ERROR);
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Step 5: directory, session.json, heartbeat.csv, module streams, session_start
	protected void OpenSession()
	{
		string failure;
		if (!MakeSessionDir(failure))
		{
			Refuse(failure);
			return;
		}

		BaseWorld world = GetGame().GetWorld();
		m_StartTimestamp = world.GetTimestamp();
		m_bHasStartTimestamp = true;
		m_iStartUnix = System.GetUnixTime();
		m_iPhysicsStep = -1;
		m_iNextHeartbeatMs = HeartbeatPeriodMs();
		m_ePhase = DC_ESessionPhase.CAPTURING;

		Print(LOG + "START session " + m_sSessionId + " dir=" + m_sSessionDir, LogLevel.NORMAL);
		if (!WriteSessionJson(world))
		{
			Seal(m_sFailureReason, false);
			return;
		}

		if (!WriteHeartbeatHeader())
		{
			Seal(m_sFailureReason, false);
			return;
		}

		foreach (DC_SessionModule module : m_aModules)
		{
			if (!module.OnSessionOpen(this))
			{
				Fail(END_SCRIPT_ERROR, module.ClassName() + " could not open its streams");
				break;
			}
		}

		if (!m_sFailureReason.IsEmpty())
		{
			Seal(m_sFailureReason, false);
			return;
		}

		// Fault case: a directory where the second ctl chunk must go, so that chunk's OpenFile or Write fails
		if (m_sFaultCase == FAULT_CHUNK_DIR_BLOCKED)
		{
			FileIO.MakeDirectory(m_sSessionDir + "/ctl/chunk_00001.csv");
			Print(LOG + "WARNING: fault case chunk_dir_blocked: ctl/chunk_00001.csv is a directory, the second ctl chunk must fail with write_fail", LogLevel.WARNING);
		}

		LogEvent("session_start", "trial=" + m_sTrialId);
	}

	//------------------------------------------------------------------------------------------------
	//! Builds the session id and creates the directory; refuses an existing one (nothing written)
	protected bool MakeSessionDir(out string failure)
	{
		m_sExeKind = GetExeKind();
		m_sInstance = GetInstanceTag();
		string utcCompact;
		m_sCreatedUtc = FormatUtc(utcCompact);

		RandomGenerator generator = new RandomGenerator();
		generator.SetSeed(System.GetTickCount());
		int randomValue = generator.RandInt(0, 65536);
		string rand4 = randomValue.ToString(4, true);
		rand4.ToLower();

		m_sSessionId = m_sMapId + "_" + m_sTrialId + "_" + utcCompact + "_" + m_sExeKind + "_" + m_sInstance + "_" + rand4;
		m_sSessionDir = SESSION_ROOT + "/" + m_sSessionId;

		// Fault case: the directory exists before the check, as after an id collision
		if (m_sFaultCase == FAULT_SESSION_DIR_EXISTS)
		{
			FileIO.MakeDirectory(SESSION_ROOT);
			FileIO.MakeDirectory(m_sSessionDir);
			Print(LOG + "WARNING: fault case session_dir_exists: created " + m_sSessionDir + " before the check", LogLevel.WARNING);
		}

		if (FileIO.FileExists(m_sSessionDir))
		{
			failure = "session directory " + m_sSessionDir + " already exists: nothing written";
			return false;
		}

		if (!FileIO.FileExists(SESSION_ROOT))
			FileIO.MakeDirectory(SESSION_ROOT);

		if (!FileIO.MakeDirectory(m_sSessionDir) || !FileIO.FileExists(m_sSessionDir))
		{
			failure = "cannot create the session directory " + m_sSessionDir;
			return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! session.json (dc-2.0 section 10.1), written once; the modules fill rig.* and control.*
	protected bool WriteSessionJson(notnull BaseWorld world)
	{
		ArmaReforgerScripted game = GetGame();
		DC_JsonWriter json = new DC_JsonWriter();
		json.BeginObject(string.Empty);
		json.AddString("contract_version", CONTRACT_VERSION);
		json.AddString("session_id", m_sSessionId);
		json.AddString("created_utc", m_sCreatedUtc);

		json.BeginObject("start");
		json.AddInt("unix_s", m_iStartUnix);
		json.AddString("local_hms", FormatLocalHms());
		int engineWorldMs = world.GetWorldTime();
		json.AddInt("engine_world_ms", engineWorldMs);
		json.AddInt("render_frame", world.GetFrameNumber());
		json.EndObject();

		json.BeginObject("map");
		json.AddString("map_id", m_sMapId);
		json.AddString("terrain", m_sTerrain);
		json.AddString("world_file", game.GetWorldFile());
		json.AddString("root_world", m_sRootWorld);
		json.BeginObject("fingerprint");
		json.AddFloat("size_m", m_fMeasuredSizeM, 3);
		json.AddFloat("latitude_deg", m_fMeasuredLatitudeDeg, 6);
		json.AddBool("ok", true);
		json.EndObject();
		json.EndObject();

		json.BeginObject("trial");
		json.AddString("trial_id", m_sTrialId);
		json.AddString("source", m_sTrialSource);
		json.BeginObject("conf");
		json.AddString("trial_id", m_sTrialId);
		json.AddFloat("duration_s", m_fDurationS, 3);
		json.AddFloat("warmup_s", m_fWarmupS, 3);
		json.AddInt("seed", m_iTrialSeed);
		json.AddString("route", m_sTrialRoute);
		json.AddBool("images", m_bTrialImages);
		json.EndObject();
		json.AddInt("seed", m_iTrialSeed);
		json.AddFloat("warmup_s", m_fWarmupS, 3);
		json.AddFloat("duration_s", m_fDurationS, 3);
		json.EndObject();

		json.BeginObject("exe");
		json.AddString("kind", m_sExeKind);
		json.AddString("instance", m_sInstance);
		json.BeginObject("cli");
		if (m_LaunchInfo.m_bCliLaunchId)
			json.AddString("dcLaunchId", m_LaunchInfo.m_sLaunchId);

		if (!m_sCliTrial.IsEmpty())
			json.AddString("dcTrial", m_sCliTrial);

		string cliInstance;
		if (System.GetCLIParam(CLI_INSTANCE, cliInstance))
			json.AddString("dcInstance", cliInstance.Trim());

		json.EndObject();
		json.EndObject();

		json.BeginObject("engine");
		json.AddString("build_version", m_sBuildVersion);
		json.AddString("build_time", game.GetBuildTime());
		json.AddFloat("fixed_time_slice_s", m_fFixedTimeSliceS, 9);
		json.EndObject();

		json.BeginObject("pin");
		json.AddString("pin_id", m_LaunchInfo.m_sPinId);
		json.AddString("pin_sha256", m_LaunchInfo.m_sPinSha256);
		json.AddStringOrNull("build_key", m_LaunchInfo.m_sBuildKey);
		json.AddString("channel", m_LaunchInfo.m_sChannel);
		json.AddString("launch_id", m_LaunchInfo.m_sLaunchId);
		json.AddString("launch_schema", m_LaunchInfo.m_sRecordSchema);
		json.AddInt("game_buildid", m_LaunchInfo.m_iGameBuildId);
		json.AddInt("tools_buildid", m_LaunchInfo.m_iToolsBuildId);
		json.EndObject();
		json.AddString("use_class", m_LaunchInfo.m_sUseClass);

		json.BeginObject("addon");
		json.AddString("addon_id", DC_BuildInfo.ADDON_ID);
		json.AddString("addon_guid", DC_BuildInfo.ADDON_GUID);
		json.AddString("git_sha", DC_BuildInfo.GIT_SHA);
		json.AddString("git_branch", DC_BuildInfo.GIT_BRANCH);
		json.AddBool("git_dirty", DC_BuildInfo.GIT_DIRTY);
		json.AddString("content_sha256", DC_BuildInfo.CONTENT_SHA256);
		json.AddString("manifest_sha256", DC_BuildInfo.MANIFEST_SHA256);
		json.AddString("deploy_utc", DC_BuildInfo.DEPLOY_UTC);
		json.AddNull("target_pin");
		json.EndObject();

		json.BeginObject("control");
		foreach (DC_SessionModule controlModule : m_aModules)
		{
			controlModule.WriteSessionJson("control", json);
		}

		json.EndObject();

		json.BeginObject("authority");
		json.AddBool("standalone", RplSession.Mode() == RplMode.None);
		json.AddBool("rpl_ok", Replication.IsServer());
		json.EndObject();

		json.BeginObject("versions");
		json.AddString("features_version", FEATURES_VERSION);
		json.AddString("nav_version", NAV_VERSION);
		json.EndObject();

		json.BeginObject("clocks");
		json.AddInt("telemetry_every_k", m_iTelemetryEveryK);
		json.AddInt("physics_steps_per_decision", m_iStepsPerDecision);
		json.AddFloat("decision_hz", m_fImageHz, 3);
		json.AddFloat("image_hz", m_fImageHz, 3);
		json.AddInt("img_inflight_timeout_ms", m_iImgInflightTimeoutMs);
		json.AddInt("img_lead_ms", m_iImgLeadMs);
		json.AddFloat("chunk_max_s", m_fChunkMaxS, 3);
		json.EndObject();

		json.BeginObject("rig");
		string rigId;
		foreach (DC_SessionModule rigModule : m_aModules)
		{
			rigModule.WriteSessionJson("rig", json);
			string part = rigModule.GetRigIdPart();
			if (part.IsEmpty())
				continue;

			if (!rigId.IsEmpty())
				rigId += "|";

			rigId += part;
		}

		string settingsHash = WriteRenderSettings(json);
		json.AddString("render_settings_hash", settingsHash);
		json.AddString("rig_id", rigId + "|rs" + settingsHash);
		json.EndObject();

		WriteEnvironment(json, ChimeraWorld.CastFrom(world));

		json.BeginObject("launch_guard");
		json.AddString("launch_mode", m_LaunchInfo.m_sMode);
		json.AddString("record_path", m_LaunchInfo.m_sRecordPath);
		json.AddString("record_launch_id", m_LaunchInfo.m_sRecordLaunchId);
		json.AddBool("cli_seen", m_LaunchInfo.m_bCliVisible);
		json.AddBool("profile_root_ok", true);
		json.AddStringOrNull("escape_binding", m_LaunchInfo.m_sEscapeBinding);
		json.EndObject();

		// Not in the dc-2.0 field table: run-time facts behind the TBD tolerances (proposed for the freeze)
		json.BeginObject("observed");
		json.AddString("step_source", m_sStepSource);
		json.AddInt("readiness_simulate_events", m_iSimulateCount);
		json.AddInt("readiness_fixedframe_events", m_iFixedFrameCount);
		json.AddString("engine_version_expected", m_sEngineVersionExpected);
		json.AddBool("keep_logs_open", m_bKeepLogsOpen);
		json.AddBool("verify_chunk_length", m_bVerifyChunkLength);
		json.AddString("fault_case", m_sFaultCase);
		json.EndObject();

		json.EndObject();

		string failure;
		if (DC_ProbeCsv.WriteLines(m_sSessionDir + "/" + SESSION_JSON, json.GetLines(), false, failure))
			return true;

		FailWrite("session.json: " + failure);
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! rig.render_settings: every SCALAR, INTEGER, FLAGS, BOOLEAN and STRING variable of the listed engine user-settings
	//! modules (generic, as E-20). \return the hash of the dump (8 hex digits of string.Hash over its canonical text)
	protected string WriteRenderSettings(notnull DC_JsonWriter json)
	{
		string canonical;
		UserSettings settings = GetGame().GetEngineUserSettings();
		json.BeginObject("render_settings");
		array<string> names = {};
		m_sRenderSettingsModules.Split(",", names, true);
		foreach (string rawName : names)
		{
			string moduleName = rawName.Trim();
			if (moduleName.IsEmpty())
				continue;

			BaseContainer module;
			if (settings)
				module = settings.GetModule(moduleName);

			if (!module)
			{
				json.AddNull(moduleName);
				canonical += moduleName + "=missing;";
				continue;
			}

			json.BeginObject(moduleName);
			for (int i = 0, count = module.GetNumVars(); i < count; i++)
			{
				string varName = module.GetVarName(i);
				string text = AddSettingValue(json, module, varName, module.GetDataVarType(i));
				if (!text.IsEmpty())
					canonical += moduleName + "." + varName + "=" + text + ";";
			}

			json.EndObject();
		}

		json.EndObject();
		return HashText(canonical);
	}

	//------------------------------------------------------------------------------------------------
	//! One typed member; \return its text for the hash, empty when the type is not dumped
	protected string AddSettingValue(notnull DC_JsonWriter json, notnull BaseContainer module, string varName, DataVarType varType)
	{
		if (varType == DataVarType.SCALAR)
		{
			float scalar;
			if (!module.Get(varName, scalar))
				return string.Empty;

			json.AddFloat(varName, scalar, 6);
			return scalar.ToString(-1, 6);
		}

		if (varType == DataVarType.INTEGER || varType == DataVarType.FLAGS)
		{
			int integer;
			if (!module.Get(varName, integer))
				return string.Empty;

			json.AddInt(varName, integer);
			return integer.ToString();
		}

		if (varType == DataVarType.BOOLEAN)
		{
			bool flag;
			if (!module.Get(varName, flag))
				return string.Empty;

			json.AddBool(varName, flag);
			return DC_LaunchGuard.BoolText(flag);
		}

		if (varType == DataVarType.STRING)
		{
			string text;
			if (!module.Get(varName, text))
				return string.Empty;

			json.AddString(varName, text);
			return text;
		}

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! env_initial: weather state, rain, fog, wetness, time of day, date and day advance at the open
	protected void WriteEnvironment(notnull DC_JsonWriter json, ChimeraWorld world)
	{
		TimeAndWeatherManagerEntity weather;
		if (world)
			weather = world.GetTimeAndWeatherManager();

		json.BeginObject("env_initial");
		if (!weather)
		{
			json.AddString("weather_state", "unknown");
			json.EndObject();
			return;
		}

		string stateName = "unknown";
		WeatherState state = weather.GetCurrentWeatherState();
		if (state)
			stateName = state.GetStateName();

		int year;
		int month;
		int day;
		weather.GetDate(year, month, day);
		json.AddString("weather_state", stateName);
		json.AddFloat("rain", weather.GetRainIntensity(), 4);
		json.AddFloat("fog", weather.GetFogAmount(), 4);
		json.AddFloat("wetness", weather.GetCurrentWetness(), 4);
		json.AddFloat("tod_h", weather.GetTimeOfTheDay(), 4);
		json.AddString("date", string.Format("%1-%2-%3", year.ToString(4), month.ToString(2), day.ToString(2)));
		json.AddBool("day_auto_advance", weather.GetIsDayAutoAdvanced());
		json.EndObject();
	}

	//------------------------------------------------------------------------------------------------
	protected bool WriteHeartbeatHeader()
	{
		string path = m_sSessionDir + "/" + HEARTBEAT_CSV;
		array<string> header = {HEARTBEAT_HEADER};
		string failure;
		if (!m_bKeepLogsOpen)
		{
			if (DC_ProbeCsv.WriteLines(path, header, false, failure))
				return true;

			FailWrite("heartbeat.csv: " + failure);
			return false;
		}

		m_HeartbeatFile = FileIO.OpenFile(path, FileMode.WRITE);
		if (!m_HeartbeatFile)
		{
			FailWrite("heartbeat.csv: FileIO.OpenFile(" + path + ", WRITE) returned null");
			return false;
		}

		return WriteHeartbeatLine(HEARTBEAT_HEADER);
	}

	//------------------------------------------------------------------------------------------------
	//! One heartbeat row (dc-2.0 section 8): appended by one open-APPEND-write-close cycle, or through the handle kept
	//! open (TBD-3 fallback). Then the DC_STOP check
	protected void WriteHeartbeat(int worldMs)
	{
		int unixTime = System.GetUnixTime();
		int memKb = System.MemoryAllocationKB();
		float fps = System.GetFPS();
		string line = string.Format("%1,%2,%3,%4,%5,%6,%7,%8", unixTime, worldMs, m_iLastPostStep, GetRenderFrame(), fps.ToString(-1, 3), memKb, m_iImagesOk, m_iImagesFailed);
		bool written;
		if (m_bKeepLogsOpen)
		{
			written = WriteHeartbeatLine(line);
		}
		else
		{
			array<string> lines = {line};
			string failure;
			written = DC_ProbeCsv.WriteLines(m_sSessionDir + "/" + HEARTBEAT_CSV, lines, true, failure);
			if (!written)
				FailWrite("heartbeat.csv: " + failure);
		}

		if (written)
			m_iHeartbeats++;

		if (m_ePhase == DC_ESessionPhase.CAPTURING && FileIO.FileExists(STOP_FILE))
		{
			LogEvent("dc_stop", "file=DC_STOP");
			BeginEnd(END_DC_STOP);
		}
	}

	//------------------------------------------------------------------------------------------------
	protected bool WriteHeartbeatLine(string line)
	{
		string data = line + "\n";
		int expected = data.Length();
		int written = m_HeartbeatFile.Write(data);
		if (written == expected)
			return true;

		FailWrite(string.Format("heartbeat.csv: short or failed write: FileHandle.Write returned %1 of %2 bytes", written, expected));
		return false;
	}

	//------------------------------------------------------------------------------------------------
	protected int HeartbeatPeriodMs()
	{
		int periodMs = m_fHeartbeatPeriodS * 1000;
		if (periodMs < 1000)
			periodMs = 1000;

		return periodMs;
	}

	//------------------------------------------------------------------------------------------------
	protected int GetDispatchEventMask()
	{
		return EntityEvent.FRAME | EntityEvent.SIMULATE | EntityEvent.POSTSIMULATE | EntityEvent.FIXEDFRAME | EntityEvent.POSTFIXEDFRAME;
	}

	//------------------------------------------------------------------------------------------------
	override void EOnSimulate(IEntity owner, float timeSlice)
	{
		m_iSimulateCount++;
		if (m_sStepSource == STEP_SIMULATE)
			HandleStepPre(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void EOnPostSimulate(IEntity owner, float timeSlice)
	{
		m_iPostSimulateCount++;
		if (m_sStepSource == STEP_SIMULATE)
			HandleStepPost(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void EOnFixedFrame(IEntity owner, float timeSlice)
	{
		m_iFixedFrameCount++;
		if (m_sStepSource == STEP_FIXEDFRAME)
			HandleStepPre(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void EOnPostFixedFrame(IEntity owner, float timeSlice)
	{
		m_iPostFixedFrameCount++;
		if (m_sStepSource == STEP_FIXEDFRAME)
			HandleStepPost(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void EOnFrame(IEntity owner, float timeSlice)
	{
		m_iFrameCount++;
		if (m_ePhase == DC_ESessionPhase.PREPARING)
		{
			UpdateReadiness();
			return;
		}

		if (IsCapturing())
			HandleFrame(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	//! Top of a physics step: physics_step + 1, OnStepPre
	protected void HandleStepPre(float timeSlice)
	{
		if (!IsCapturing())
			return;

		if (m_iPhysicsStep >= 0 && m_iLastPostStep < m_iPhysicsStep)
			m_iPreWithoutPost++;

		m_iPhysicsStep++;
		foreach (DC_SessionModule module : m_aModules)
		{
			if (!m_sFailureReason.IsEmpty())
				break;

			module.OnStepPre(m_iPhysicsStep, timeSlice);
		}

		SealOnFailure();
	}

	//------------------------------------------------------------------------------------------------
	//! Same step after the simulation: OnStepPost (rows), warm-up and planned end, OnStepDecide. A post without its
	//! own pre call is not dispatched (a second ctl row of one step would break the key) and is counted
	protected void HandleStepPost(float timeSlice)
	{
		if (!IsCapturing())
			return;

		if (m_iPhysicsStep < 0 || m_iLastPostStep == m_iPhysicsStep)
		{
			m_iPostWithoutPre++;
			return;
		}

		int worldMs = GetWorldMs();
		if (worldMs <= m_iLastStepWorldMs)
			m_iStepsWithoutWorldAdvance++;

		m_iLastStepWorldMs = worldMs;
		m_iLastPostStep = m_iPhysicsStep;
		if (m_iPhysicsStep == 0)
			m_iStepZeroWorldMs = worldMs;

		foreach (DC_SessionModule module : m_aModules)
		{
			if (!m_sFailureReason.IsEmpty())
				break;

			module.OnStepPost(m_iPhysicsStep, timeSlice, worldMs);
		}

		if (m_sFailureReason.IsEmpty() && !m_bWarmupOver && worldMs >= m_fWarmupS * 1000)
		{
			m_bWarmupOver = true;
			LogEvent("warmup_end", string.Empty);
		}

		if (m_sFailureReason.IsEmpty() && m_ePhase == DC_ESessionPhase.CAPTURING && worldMs >= (m_fWarmupS + m_fDurationS) * 1000)
			BeginEnd(END_DURATION);

		foreach (DC_SessionModule decider : m_aModules)
		{
			if (!m_sFailureReason.IsEmpty())
				break;

			decider.OnStepDecide(m_iPhysicsStep);
		}

		SealOnFailure();
	}

	//------------------------------------------------------------------------------------------------
	//! One render frame while capturing or draining
	protected void HandleFrame(float timeSlice)
	{
		int renderFrame = GetRenderFrame();
		int worldMs = GetWorldMs();
		foreach (DC_SessionModule starter : m_aModules)
		{
			if (!m_sFailureReason.IsEmpty())
				break;

			starter.OnFrameStart(renderFrame);
		}

		foreach (DC_SessionModule module : m_aModules)
		{
			if (!m_sFailureReason.IsEmpty())
				break;

			module.OnFrame(renderFrame, timeSlice, worldMs);
		}

		if (SealOnFailure())
			return;

		foreach (DC_SessionStream stream : m_aStreams)
		{
			if (!stream.FlushIfDue(worldMs))
				FailWrite(stream.GetName() + ": " + stream.GetFailure());
		}

		if (SealOnFailure())
			return;

		if (worldMs >= m_iNextHeartbeatMs)
		{
			m_iNextHeartbeatMs = worldMs - worldMs % HeartbeatPeriodMs() + HeartbeatPeriodMs();
			WriteHeartbeat(worldMs);
			if (SealOnFailure())
				return;
		}

		if (m_ePhase == DC_ESessionPhase.DRAINING && IsDrainDone(worldMs))
			Seal(m_sEndReason, false);
	}

	//------------------------------------------------------------------------------------------------
	//! true when every module is drained, or when the in-flight timeout passed in real or in world time
	protected bool IsDrainDone(int worldMs)
	{
		if (System.GetTickCount() - m_iDrainStartTickMs >= m_iImgInflightTimeoutMs || worldMs - m_iDrainStartWorldMs >= m_iImgInflightTimeoutMs)
			return true;

		foreach (DC_SessionModule module : m_aModules)
		{
			if (!module.IsDrained())
				return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Planned end (duration, dc_stop): the modules stop and drain. Idempotent
	protected void BeginEnd(string endReason)
	{
		if (m_ePhase != DC_ESessionPhase.CAPTURING)
			return;

		m_ePhase = DC_ESessionPhase.DRAINING;
		m_bStopping = true;
		m_sEndReason = endReason;
		m_iDrainStartTickMs = System.GetTickCount();
		m_iDrainStartWorldMs = GetWorldMs();
		Print(string.Format(LOG + "session end (%1) at physics step %2: draining", endReason, m_iLastPostStep), LogLevel.NORMAL);
		foreach (DC_SessionModule module : m_aModules)
		{
			module.OnSessionStopping(endReason);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! \return true when a failure was reported and the session is now sealed
	protected bool SealOnFailure()
	{
		if (m_sFailureReason.IsEmpty() || m_ePhase == DC_ESessionPhase.CLOSED)
			return false;

		Seal(m_sFailureReason, false);
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Step 7: last module rows, the streams committed (or, after a write failure, abandoned: no further row is
	//! acknowledged; only the events row script_error is attempted), session_end.json last, then the release.
	//! teardown = world deletion: no module is called. Idempotent
	protected void Seal(string endReason, bool teardown)
	{
		if (m_ePhase == DC_ESessionPhase.CLOSED || m_ePhase == DC_ESessionPhase.IDLE)
			return;

		if (m_ePhase == DC_ESessionPhase.PREPARING)
		{
			Refuse("stopped during readiness (" + endReason + ")");
			return;
		}

		m_ePhase = DC_ESessionPhase.CLOSED;
		m_bStopping = true;

		// After a write failure no further row is acknowledged: every stream but events is dropped before the modules
		// write their last rows
		if (m_bWriteFailed)
		{
			foreach (DC_SessionStream droppedStream : m_aStreams)
			{
				if (droppedStream.GetName() != "events" || droppedStream.HasFailed())
					droppedStream.Abandon();
			}
		}

		if (!teardown)
		{
			foreach (DC_SessionModule module : m_aModules)
			{
				module.OnSessionClosing(endReason);
			}
		}

		if (m_bWriteFailed)
		{
			// Only the events row script_error is still attempted
			LogEvent("script_error", "reason=write_fail");
			DC_SessionStream events = FindStream("events");
			if (events && !events.HasFailed())
				events.Close();
		}
		else
		{
			if (endReason == END_SCRIPT_ERROR)
				LogEvent("script_error", "reason=other");

			// A commit that fails here stops the acknowledgement too: later streams are dropped, and the events row
			// script_error is attempted while the events stream is still open
			foreach (DC_SessionStream stream : m_aStreams)
			{
				bool isEvents = stream.GetName() == "events";
				if (m_bWriteFailed && !isEvents)
				{
					stream.Abandon();
					continue;
				}

				if (m_bWriteFailed && isEvents)
					LogEvent("script_error", "reason=write_fail");

				if (!stream.Close())
				{
					m_bWriteFailed = true;
					if (m_sFailureDetail.IsEmpty())
						m_sFailureDetail = stream.GetName() + ": " + stream.GetFailure();
				}
			}
		}

		if (m_bWriteFailed)
			endReason = END_WRITE_FAIL;

		string status = "ok";
		if (endReason != END_DURATION && endReason != END_STOP && endReason != END_DC_STOP)
			status = "failed";

		bool endWritten = WriteSessionEnd(status, endReason);
		if (m_HeartbeatFile)
		{
			m_HeartbeatFile.Close();
			m_HeartbeatFile = null;
		}

		string line = string.Format(LOG + "END session %1 status=%2 end_reason=%3 physics_steps=%4 frames=%5 heartbeats=%6 images issued %7 ok %8 failed %9", m_sSessionId, status, endReason, m_iLastPostStep + 1, m_iFrameCount, m_iHeartbeats, m_iImagesIssued, m_iImagesOk, m_iImagesFailed);
		if (!m_sFailureDetail.IsEmpty())
			line += " (" + m_sFailureDetail + ")";

		if (!endWritten)
			line += " - session_end.json could NOT be written: the session is unsealed";

		if (status == "ok" && endWritten)
			Print(line, LogLevel.NORMAL);
		else
			Print(line, LogLevel.ERROR);

		Release(teardown);

#ifndef WORKBENCH
		if (!teardown && endReason == END_DC_STOP && m_bCloseOnDcStop)
			GetGame().RequestClose();
#endif
	}

	//------------------------------------------------------------------------------------------------
	//! session_end.json (dc-2.0 section 10.2), written last. \return false when it could not be written
	protected bool WriteSessionEnd(string status, string endReason)
	{
		DC_JsonWriter json = new DC_JsonWriter();
		json.BeginObject(string.Empty);
		json.AddString("contract_version", CONTRACT_VERSION);
		json.AddString("session_id", m_sSessionId);
		json.AddString("status", status);
		json.AddString("end_reason", endReason);

		string utcCompact;
		json.BeginObject("end");
		json.AddInt("unix_s", System.GetUnixTime());
		json.AddString("utc", FormatUtc(utcCompact));
		json.AddInt("world_ms", GetWorldMs());
		json.AddInt("physics_step", GetStreamLastKey("ctl", m_iLastPostStep));
		json.AddInt("render_frame", GetStreamLastKey("hf", GetRenderFrame()));
		json.EndObject();

		json.BeginObject("pin");
		json.AddString("pin_id", m_LaunchInfo.m_sPinId);
		json.AddStringOrNull("build_key", m_LaunchInfo.m_sBuildKey);
		json.EndObject();

		json.BeginObject("engine");
		json.AddString("build_version", GetGame().GetBuildVersion());
		json.EndObject();

		json.BeginObject("counts");
		int episodes;
		foreach (DC_SessionStream stream : m_aStreams)
		{
			json.BeginObject(stream.GetName());
			json.AddInt("rows", stream.GetCommittedRows());
			json.AddInt("chunks", stream.GetCommittedChunks());
			if (stream.GetCommittedRows() > 0)
				json.AddInt("last_key", stream.GetLastCommittedKey());
			else
				json.AddNull("last_key");

			json.EndObject();
			if (stream.GetName() == "episodes")
				episodes = stream.GetCommittedRows();
		}

		json.EndObject();

		json.BeginObject("images");
		json.AddInt("requested", m_iImagesRequested);
		json.AddInt("issued", m_iImagesIssued);
		json.AddInt("skipped_busy", m_iImagesSkippedBusy);
		json.AddInt("make_failed", m_iImagesMakeFailed);
		json.AddInt("timed_out", m_iImagesTimedOut);
		json.AddInt("pending_at_end", m_iImagesPending);
		json.EndObject();

		json.AddInt("episodes", episodes);
		json.AddInt("heartbeats", m_iHeartbeats);

		// Not in the dc-2.0 field table: run-time facts behind the TBD tolerances (proposed for the freeze)
		NoteObserved("post_without_pre", m_iPostWithoutPre.ToString());
		NoteObserved("pre_without_post", m_iPreWithoutPost.ToString());
		NoteObserved("steps_without_world_advance", m_iStepsWithoutWorldAdvance.ToString());
		NoteObserved("failure", m_sFailureDetail);
		json.BeginObject("observed");
		for (int i = 0, count = m_aObservedKeys.Count(); i < count; i++)
		{
			json.AddString(m_aObservedKeys[i], m_aObservedValues[i]);
		}

		json.EndObject();
		json.EndObject();

		string failure;
		if (DC_ProbeCsv.WriteLines(m_sSessionDir + "/" + SESSION_END_JSON, json.GetLines(), false, failure))
			return true;

		Print(LOG + "ERROR: session_end.json: " + failure, LogLevel.ERROR);
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Last committed key of a registered stream (ctl, hf), else fallback
	protected int GetStreamLastKey(string name, int fallback)
	{
		DC_SessionStream stream = FindStream(name);
		if (!stream)
			return fallback;

		if (stream.GetCommittedRows() == 0)
			return -1;

		return stream.GetLastCommittedKey();
	}

	//------------------------------------------------------------------------------------------------
	//! A refusal before the directory exists: one ERROR line, nothing written, modules released
	protected void Refuse(string reason)
	{
		Print(LOG + "ERROR: " + reason + " - no session, nothing written", LogLevel.ERROR);
		m_ePhase = DC_ESessionPhase.CLOSED;
		Release(false);
	}

	//------------------------------------------------------------------------------------------------
	//! Modules released (not on world teardown: no other component is called during deletion, each module drops its
	//! references in its own OnDelete), event mask cleared, FallbackStart removed. Idempotent
	protected void Release(bool teardown)
	{
		if (!teardown)
		{
			foreach (DC_SessionModule module : m_aModules)
			{
				if (module)
					module.OnSessionRelease(false);
			}
		}

		m_aModules.Clear();
		ArmaReforgerScripted game = GetGame();
		if (game)
		{
			ScriptCallQueue queue = game.GetCallqueue();
			if (queue)
				queue.Remove(FallbackStart);
		}

		if (!m_bMaskSet)
			return;

		m_bMaskSet = false;
		IEntity owner = GetOwner();
		if (owner)
			ClearEventMask(owner, GetDispatchEventMask());
	}

	//------------------------------------------------------------------------------------------------
	//! Called on every machine when the game mode ends (SCR_BaseGameMode.OnGameStateChanged, POSTGAME)
	override void OnGameModeEnd(SCR_GameModeEndData data)
	{
		super.OnGameModeEnd(data);
		StopRun();
	}

	//------------------------------------------------------------------------------------------------
	//! Called on every machine when the world ends, before the entities are deleted
	override void OnGameEnd()
	{
		super.OnGameEnd();
		StopRun();
	}

	//------------------------------------------------------------------------------------------------
	override void OnDelete(IEntity owner)
	{
		// World teardown: no other component is called (the DD-1 rule); the streams the session owns are committed
		if (IsCapturing())
			Seal(GetStopReason(), true);
		else
			Release(true);

		super.OnDelete(owner);
	}

	//------------------------------------------------------------------------------------------------
	//! Play stopped: seal at once (no frame follows to drain in), or end the readiness. Idempotent
	protected void StopRun()
	{
		if (m_ePhase == DC_ESessionPhase.PREPARING)
		{
			Refuse("Play stopped during readiness");
			return;
		}

		if (IsCapturing())
		{
			Seal(GetStopReason(), false);
			return;
		}

		Release(false);
	}

	//------------------------------------------------------------------------------------------------
	//! stop, or the planned end reason when the Play stops while the session drains
	protected string GetStopReason()
	{
		if (m_sEndReason.IsEmpty())
			return END_STOP;

		return m_sEndReason;
	}

	//------------------------------------------------------------------------------------------------
	// Module API
	//------------------------------------------------------------------------------------------------
	//! A module refuses readiness (before the directory exists): ERROR, no session
	void RefuseReadiness(string reason)
	{
		if (m_sReadinessRefusal.IsEmpty())
			m_sReadinessRefusal = reason;
	}

	//------------------------------------------------------------------------------------------------
	//! Ends the session with an error end reason (zero_ego, ego_lost, camera_lost, script_error:other) after the
	//! current dispatch. The first failure is kept
	void Fail(string endReason, string detail)
	{
		if (!m_sFailureReason.IsEmpty() || !IsCapturing())
			return;

		m_sFailureReason = endReason;
		m_sFailureDetail = detail;
		Print(LOG + "ERROR: " + endReason + ": " + detail, LogLevel.ERROR);
	}

	//------------------------------------------------------------------------------------------------
	//! A short or failed write (chunk, commit line, heartbeat row, session.json): the session stops with
	//! script_error:write_fail and acknowledges no further row (dc-2.0 section 8)
	void FailWrite(string detail)
	{
		m_bWriteFailed = true;
		if (m_sFailureReason.IsEmpty())
		{
			m_sFailureReason = END_WRITE_FAIL;
			m_sFailureDetail = detail;
		}

		Print(LOG + "ERROR: write_fail: " + detail, LogLevel.ERROR);
	}

	//------------------------------------------------------------------------------------------------
	//! true once a write failed: modules write no further row
	bool HasWriteFailure()
	{
		return m_bWriteFailed;
	}

	//------------------------------------------------------------------------------------------------
	//! The session owns stream writers: it commits them at the seal, counts them in session_end.json and checks them
	//! for failed writes every frame
	void RegisterStream(notnull DC_SessionStream stream)
	{
		m_aStreams.Insert(stream);
	}

	//------------------------------------------------------------------------------------------------
	DC_SessionStream FindStream(string name)
	{
		foreach (DC_SessionStream stream : m_aStreams)
		{
			if (stream.GetName() == name)
				return stream;
		}

		return null;
	}

	//------------------------------------------------------------------------------------------------
	//! One events row (dc-2.0 9.5) into the registered events stream; the line is always printed. kind must be a
	//! contract event kind; payload is 'key=value' pairs joined by ';' (',', '"', CR and LF become '_')
	bool LogEvent(string kind, string payload)
	{
		string clean = payload;
		clean.Replace(",", "_");
		clean.Replace("\"", "_");
		clean.Replace("\r", "_");
		clean.Replace("\n", "_");
		if (s_aEventKinds.Find(kind) < 0)
		{
			Print(LOG + "ERROR: event kind '" + kind + "' is not a dc-2.0 event kind - not recorded", LogLevel.ERROR);
			return false;
		}

		Print(string.Format(LOG + "event %1 %2 (step %3, episode %4)", kind, clean, m_iPhysicsStep, m_iEpisodeId), LogLevel.NORMAL);
		DC_SessionStream events = FindStream("events");
		if (!events || !m_bHasStartTimestamp)
			return false;

		int worldMs = GetWorldMs();
		m_EventRow.Clear();
		m_EventRow.AddInt(m_iEventSeq);
		m_EventRow.AddInt(m_iEpisodeId);
		m_EventRow.AddInt(m_iPhysicsStep);
		m_EventRow.AddInt(GetRenderFrame());
		m_EventRow.AddInt(worldMs);
		m_EventRow.AddString(kind);
		m_EventRow.AddInt(m_iEgoSlotId);
		m_EventRow.AddString(clean);
		if (!events.AddRow(m_EventRow, m_iEventSeq, worldMs))
		{
			FailWrite("events: " + events.GetFailure());
			return false;
		}

		m_iEventSeq++;
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The events columns of dc-2.0 9.5, in order
	static void AddEventColumns(notnull array<string> columns)
	{
		columns.Insert("event_seq");
		columns.Insert("episode_id");
		columns.Insert("physics_step");
		columns.Insert("render_frame");
		columns.Insert("world_time_ms");
		columns.Insert("kind");
		columns.Insert("veh_slot_id");
		columns.Insert("payload");
	}

	//------------------------------------------------------------------------------------------------
	//! A run-time fact for session_end.json observed (the last value of a key wins)
	void NoteObserved(string key, string value)
	{
		int index = m_aObservedKeys.Find(key);
		if (index >= 0)
		{
			m_aObservedValues[index] = value;
			return;
		}

		m_aObservedKeys.Insert(key);
		m_aObservedValues.Insert(value);
	}

	//------------------------------------------------------------------------------------------------
	//! The ego of the session (one for its whole life): entity, slot id, IEntity.GetID() token, prefab
	void SetEgo(IEntity ego, int slotId, string entityId, string prefab)
	{
		m_Ego = ego;
		m_iEgoSlotId = slotId;
		m_sEgoEntityId = entityId;
		m_sEgoPrefab = prefab;
	}

	//------------------------------------------------------------------------------------------------
	//! Image slot counts (session_end.json images): issued, skipped_busy or make_failed
	void CountImageSlot(string issueStatus)
	{
		m_iImagesRequested++;
		if (issueStatus == "issued")
			m_iImagesIssued++;
		else if (issueStatus == "skipped_busy")
			m_iImagesSkippedBusy++;
		else if (issueStatus == "make_failed")
			m_iImagesMakeFailed++;
	}

	//------------------------------------------------------------------------------------------------
	//! A request resolved: its file was seen (img_ok) or it timed out (img_fail, timed_out)
	void CountImageResolved(bool fileSeen)
	{
		if (fileSeen)
		{
			m_iImagesOk++;
			return;
		}

		m_iImagesFailed++;
		m_iImagesTimedOut++;
	}

	//------------------------------------------------------------------------------------------------
	//! Requests still unresolved when the session sealed (images.pending_at_end)
	void SetImagesPending(int pending)
	{
		m_iImagesPending = pending;
	}

	//------------------------------------------------------------------------------------------------
	void SetEpisodeId(int episodeId)
	{
		m_iEpisodeId = episodeId;
	}

	//------------------------------------------------------------------------------------------------
	int GetEpisodeId()
	{
		return m_iEpisodeId;
	}

	//------------------------------------------------------------------------------------------------
	IEntity GetEgo()
	{
		return m_Ego;
	}

	//------------------------------------------------------------------------------------------------
	int GetEgoSlotId()
	{
		return m_iEgoSlotId;
	}

	//------------------------------------------------------------------------------------------------
	string GetEgoEntityId()
	{
		return m_sEgoEntityId;
	}

	//------------------------------------------------------------------------------------------------
	string GetEgoPrefab()
	{
		return m_sEgoPrefab;
	}

	//------------------------------------------------------------------------------------------------
	//! true while rows are written (capturing or draining)
	bool IsCapturing()
	{
		return m_ePhase == DC_ESessionPhase.CAPTURING || m_ePhase == DC_ESessionPhase.DRAINING;
	}

	//------------------------------------------------------------------------------------------------
	//! true once the planned end began or the session is sealed: no new episode or image request
	bool IsStopping()
	{
		return m_bStopping;
	}

	//------------------------------------------------------------------------------------------------
	bool IsWarmupOver()
	{
		return m_bWarmupOver;
	}

	//------------------------------------------------------------------------------------------------
	//! world_time_ms: WorldTimestamp.DiffMilliseconds against the open, whole ms; 0 before the open
	int GetWorldMs()
	{
		if (!m_bHasStartTimestamp)
			return 0;

		BaseWorld world = GetGame().GetWorld();
		if (!world)
			return m_iLastStepWorldMs;

		WorldTimestamp now = world.GetTimestamp();
		int worldMs = Math.Floor(now.DiffMilliseconds(m_StartTimestamp));
		return worldMs;
	}

	//------------------------------------------------------------------------------------------------
	//! BaseWorld.GetFrameNumber(), -1 without a world
	int GetRenderFrame()
	{
		ArmaReforgerScripted game = GetGame();
		if (!game)
			return -1;

		BaseWorld world = game.GetWorld();
		if (!world)
			return -1;

		return world.GetFrameNumber();
	}

	//------------------------------------------------------------------------------------------------
	//! ChimeraWorld.IsGameTimePaused()
	bool IsPaused()
	{
		ChimeraWorld world = ChimeraWorld.CastFrom(GetGame().GetWorld());
		if (!world)
			return false;

		return world.IsGameTimePaused();
	}

	//------------------------------------------------------------------------------------------------
	//! The step being dispatched (or the last one), -1 before step 0
	int GetPhysicsStep()
	{
		return m_iPhysicsStep;
	}

	//------------------------------------------------------------------------------------------------
	//! hf.last_physics_step: the step of the most recent OnStepPost, -1 before the first
	int GetLastPostStep()
	{
		return m_iLastPostStep;
	}

	//------------------------------------------------------------------------------------------------
	//! world_time_ms of physics step 0 (the image grid anchor), -1 before it
	int GetStepZeroWorldMs()
	{
		return m_iStepZeroWorldMs;
	}

	//------------------------------------------------------------------------------------------------
	string GetSessionId()
	{
		return m_sSessionId;
	}

	//------------------------------------------------------------------------------------------------
	//! $profile:DC_Sessions/<session_id>
	string GetSessionDir()
	{
		return m_sSessionDir;
	}

	//------------------------------------------------------------------------------------------------
	float GetFixedTimeSliceS()
	{
		return m_fFixedTimeSliceS;
	}

	//------------------------------------------------------------------------------------------------
	//! clocks.physics_steps_per_decision (TBD-1)
	int GetStepsPerDecision()
	{
		return m_iStepsPerDecision;
	}

	//------------------------------------------------------------------------------------------------
	//! clocks.telemetry_every_k (TBD-1)
	int GetTelemetryEveryK()
	{
		return m_iTelemetryEveryK;
	}

	//------------------------------------------------------------------------------------------------
	float GetImageHz()
	{
		return m_fImageHz;
	}

	//------------------------------------------------------------------------------------------------
	int GetImgInflightTimeoutMs()
	{
		return m_iImgInflightTimeoutMs;
	}

	//------------------------------------------------------------------------------------------------
	int GetImgLeadMs()
	{
		return m_iImgLeadMs;
	}

	//------------------------------------------------------------------------------------------------
	//! clocks.chunk_max_s in ms
	int GetChunkMaxMs()
	{
		int chunkMs = m_fChunkMaxS * 1000;
		return chunkMs;
	}

	//------------------------------------------------------------------------------------------------
	bool GetKeepLogsOpen()
	{
		return m_bKeepLogsOpen;
	}

	//------------------------------------------------------------------------------------------------
	bool GetVerifyChunkLength()
	{
		return m_bVerifyChunkLength;
	}

	//------------------------------------------------------------------------------------------------
	string GetTrialId()
	{
		return m_sTrialId;
	}

	//------------------------------------------------------------------------------------------------
	//! The trial conf's route (empty without one); DD-7-min's driver reads its route file from it
	string GetTrialRoute()
	{
		return m_sTrialRoute;
	}

	//------------------------------------------------------------------------------------------------
	int GetTrialSeed()
	{
		return m_iTrialSeed;
	}

	//------------------------------------------------------------------------------------------------
	//! false for a telemetry-only trial (T0a: "images": false in the conf)
	bool GetTrialImages()
	{
		return m_bTrialImages;
	}

	//------------------------------------------------------------------------------------------------
	string GetMapId()
	{
		return m_sMapId;
	}

	//------------------------------------------------------------------------------------------------
	string GetPinId()
	{
		if (!m_LaunchInfo)
			return string.Empty;

		return m_LaunchInfo.m_sPinId;
	}

	//------------------------------------------------------------------------------------------------
	//! pin.build_key (empty with a dd-launch/1 record)
	string GetBuildKey()
	{
		if (!m_LaunchInfo)
			return string.Empty;

		return m_LaunchInfo.m_sBuildKey;
	}

	//------------------------------------------------------------------------------------------------
	string GetFaultCase()
	{
		return m_sFaultCase;
	}

	//------------------------------------------------------------------------------------------------
	//! "wb" in Workbench; outside, "cld" in a diag build (ENABLE_DIAG) and "cl" otherwise
	static string GetExeKind()
	{
		string kind = "cl";
#ifdef ENABLE_DIAG
		kind = "cld";
#endif
#ifdef WORKBENCH
		kind = "wb";
#endif
		return kind;
	}

	//------------------------------------------------------------------------------------------------
	//! "i00" in Workbench; outside, "i" + the two digits of -dcInstance (i00 without it)
	static string GetInstanceTag()
	{
#ifdef WORKBENCH
		return "i00";
#else
		string value;
		if (!System.GetCLIParam(CLI_INSTANCE, value))
			return "i00";

		value = value.Trim();
		if (value.Length() != 2 || !value.IsDigitAt(0) || !value.IsDigitAt(1))
		{
			Print(LOG + "WARNING: -dcInstance '" + value + "' is not two digits - using i00", LogLevel.WARNING);
			return "i00";
		}

		return "i" + value;
#endif
	}

	//------------------------------------------------------------------------------------------------
	//! Trial ids are 1-16 characters of [A-Za-z0-9-] (the session id rule: no underscore)
	static bool IsValidTrialId(string trialId)
	{
		int length = trialId.Length();
		if (length < 1 || length > 16)
			return false;

		for (int i = 0; i < length; i++)
		{
			int c = trialId.ToAscii(i);
			bool digit = c >= 48 && c <= 57;		// 0-9
			bool upper = c >= 65 && c <= 90;		// A-Z
			bool lower = c >= 97 && c <= 122;		// a-z
			if (!digit && !upper && !lower && c != 45)	// 45 = '-'
				return false;
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! "YYYY-MM-DDTHH:MM:SSZ" from the UTC wall clock; compact gets "YYYYMMDDTHHMMSSZ" of the same reading. The date
	//! is read on both sides of the time, so a read across midnight takes the time again with the later date
	static string FormatUtc(out string compact)
	{
		int year;
		int month;
		int day;
		int hour;
		int minute;
		int second;
		System.GetYearMonthDayUTC(year, month, day);
		System.GetHourMinuteSecondUTC(hour, minute, second);

		int yearAfter;
		int monthAfter;
		int dayAfter;
		System.GetYearMonthDayUTC(yearAfter, monthAfter, dayAfter);
		if (yearAfter != year || monthAfter != month || dayAfter != day)
		{
			year = yearAfter;
			month = monthAfter;
			day = dayAfter;
			System.GetHourMinuteSecondUTC(hour, minute, second);
		}

		compact = year.ToString(4) + month.ToString(2) + day.ToString(2) + "T" + hour.ToString(2) + minute.ToString(2) + second.ToString(2) + "Z";
		return string.Format("%1-%2-%3T%4:%5:%6Z", year.ToString(4), month.ToString(2), day.ToString(2), hour.ToString(2), minute.ToString(2), second.ToString(2));
	}

	//------------------------------------------------------------------------------------------------
	//! Local "hh:mm:ss" (session.json start.local_hms, for console.log slicing)
	static string FormatLocalHms()
	{
		int hour;
		int minute;
		int second;
		System.GetHourMinuteSecond(hour, minute, second);
		return hour.ToString(2) + ":" + minute.ToString(2) + ":" + second.ToString(2);
	}

	//------------------------------------------------------------------------------------------------
	//! 8 lower-case hex digits of string.Hash(), byte by byte (no sign issue)
	static string HashText(string text)
	{
		int hash = text.Hash();
		string hex;
		for (int shift = 24; shift >= 0; shift -= 8)
		{
			int byteValue = (hash >> shift) & 255;
			hex += byteValue.ToString(2, true);
		}

		hex.ToLower();
		return hex;
	}
}
