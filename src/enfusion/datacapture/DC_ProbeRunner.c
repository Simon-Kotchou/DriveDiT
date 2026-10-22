// ============================================================================
// DC_ProbeRunner - T0-ENG probe runner on the DC_Probes_Arland game mode (DD-4-probe)
// ============================================================================
//
// Runs the T0-ENG core probes (r2 4.4) one Play at a time. It sits on GameMode_DCProbes in
// worlds/DataCapture/DC_Probes_Arland next to the probe components (DC_ProbeComponent subclasses: DC_ProbeEngine,
// DC_ProbeSequences, DC_ProbeImages, DC_ProbeMarkers, DC_RoadDump). DC_Arland and its dc-1.0 producers are not
// involved. A probe never sets its own event mask: the runner receives every engine event and dispatches it.
//
// WHAT IT DOES (authority only, play mode only; nothing at all in edit mode)
//   1. Logs the E-11 facts in one line: is -profile visible, is -dcLaunchId visible, $profile (Workbench), exe.
//   2. Runs DC_LaunchGuard.Evaluate. Refused: one ERROR line and nothing is written, not even a directory.
//   3. Builds the plan: -dcProbePlan <ids> when the command line is visible, else m_sPlan. An id without a probe
//      component is skipped with a WARNING. DC_Ego (m_sEgoName) is looked up once; probes that need it refuse
//      without it.
//   4. Sets FRAME | SIMULATE | POSTSIMULATE | FIXEDFRAME | POSTFIXEDFRAME on the game mode entity and starts the
//      concurrent probes (IsConcurrent: only e10) at once.
//   5. After a warm-up of m_fWarmupS world seconds decides the physics-step source: "simulate" if any EOnSimulate
//      arrived, else "fixedframe" (WARNING) if any EOnFixedFrame arrived, else "none" (ERROR: probes that need steps
//      refuse to start). Whether SIMULATE fires on an entity without physics is unknown (PROBE_DESIGN.md 11.4).
//   6. Runs the exclusive probes one after another, m_fGapS world seconds apart. A probe stops when its context saw
//      a failed write (write_fail), when IsProbeDone says so (done) or after GetProbeTimeoutS (timeout).
//   7. Plan finished: stops the concurrent probes (done), clears the event mask, logs "plan complete".
//
// DISPATCH: EOnSimulate always counts; while the step source is "simulate" or undecided it also increments the
//   physics step and calls OnProbeSimulate on the running exclusive probe, then on the concurrent ones.
//   EOnPostSimulate does the same with the same step number (a post without a pre is passed with the last number and
//   shows up in E-10). In the "fixedframe" fallback EOnFixedFrame / EOnPostFixedFrame do it instead. EOnFrame calls
//   OnProbeFrame (exclusive first) and then sequences the plan. No CallLater clock is used for any of this: the only
//   queued call is FallbackStart.
//
// OUTPUT (PROBE_DESIGN.md 4.5): $profile:DC_Probes/<probe>/<launch_id>/play_NN/ with run_header.txt (written before
//   OnProbeStart), the probe's own files and run_footer.txt (written at the stop; absent after a crash or a kill).
//   Console markers: "[DC_Probe <id>] START play_dir=<dir>" and "[DC_Probe <id>] END reason=<r>".
//
// LIFECYCLE (as SCR_MLDataCaptureDirector): the authority's first EOnFrame of the game mode calls OnGameModeStart on
//   every SCR_BaseGameModeComponent; FallbackStart starts the run m_iFallbackStartMs of world time after OnPostInit
//   if that never happened. Both go through the idempotent StartRun. OnGameModeEnd / OnGameEnd stop the run with
//   reason play_end: the running probes get OnProbeStop and restore what they changed. OnDelete (world teardown)
//   follows the DD-1 rule "do not touch another component during deletion" (SCR_MLDataCaptureDirector.c:449-454):
//   it calls no probe component, it only closes the files the contexts own and writes the footers (status stopped,
//   reason delete). Each probe component stops itself in its own OnDelete (DC_ProbeComponent base class).
//
// FAILURE HANDLING: a probe that cannot get a play directory is skipped (ERROR). A probe whose header cannot be
//   written, whose ego or step source is missing, or whose OnProbeStart returns false ends at once with footer
//   status failed, reason start_failed (OnProbeStop(id, "error") lets it undo a partial start). The writer's first
//   failure message (requested and returned byte counts) is logged and written to the footer note.
// ============================================================================

//------------------------------------------------------------------------------------------------
//! Base of every probe component on the probe game mode. DC_ProbeRunner discovers them with
//! FindComponents(DC_ProbeComponent, ...) and drives them; a probe never sets its own event mask.
class DC_ProbeComponentClass : ScriptComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeComponent : ScriptComponent
{
	protected ref array<string> m_aActiveProbeIds = {};	//!< kept by MarkProbeActive

	//------------------------------------------------------------------------------------------------
	//! Probe ids this component implements, for example {"e01", "e02a"}
	void GetProbeIds(notnull array<string> outIds)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! true: started with the plan and stopped with it, next to the exclusive probes (only "e10")
	bool IsConcurrent(string probeId)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! true: the runner refuses to start the probe when DC_Ego is missing
	bool NeedsEgo(string probeId)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! true: the runner refuses to start the probe when no step source was found in the warm-up
	bool NeedsSteps(string probeId)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Hard limit in world seconds; the runner then stops the probe with reason "timeout"
	float GetProbeTimeoutS(string probeId)
	{
		return 600;
	}

	//------------------------------------------------------------------------------------------------
	//! key=value strings, written as params.<key>=<value> into the run header (every attribute of the probe)
	void GetProbeParams(string probeId, notnull array<string> outKeyValues)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Called once after the play dir and run_header.txt exist. false = probe failed to start (footer status failed)
	bool OnProbeStart(string probeId, notnull DC_ProbeContext ctx)
	{
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Physics fixed step, before the simulation (EOnSimulate, or EOnFixedFrame in the fallback)
	void OnProbeSimulate(string probeId, float timeSlice, int physicsStep)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Same step after the simulation (EOnPostSimulate, or EOnPostFixedFrame)
	void OnProbePostSimulate(string probeId, float timeSlice, int physicsStep)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Every render frame (EOnFrame)
	void OnProbeFrame(string probeId, float timeSlice, int renderFrame)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Polled every frame for the running exclusive probe; true makes the runner stop it with reason "done"
	bool IsProbeDone(string probeId)
	{
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent. reason: done | timeout | write_fail | error | play_end | delete.
	//! "delete" = world teardown (OnDelete): close files only, touch no other entity, camera or setting
	void OnProbeStop(string probeId, string reason)
	{
	}

	//------------------------------------------------------------------------------------------------
	//! Kept by the base class, set by the runner right before OnProbeStart / right after OnProbeStop
	void MarkProbeActive(string probeId, bool active)
	{
		int index = m_aActiveProbeIds.Find(probeId);
		if (active && index < 0)
			m_aActiveProbeIds.Insert(probeId);
		else if (!active && index >= 0)
			m_aActiveProbeIds.Remove(index);
	}

	//------------------------------------------------------------------------------------------------
	bool IsProbeActive(string probeId)
	{
		return m_aActiveProbeIds.Find(probeId) >= 0;
	}

	//------------------------------------------------------------------------------------------------
	//! OnProbeStop(id, "delete") for every still-active id, then super.OnDelete. A probe that overrides OnDelete
	//! calls super.OnDelete(owner) last.
	override void OnDelete(IEntity owner)
	{
		array<string> activeIds = {};
		foreach (string probeId : m_aActiveProbeIds)
		{
			activeIds.Insert(probeId);
		}

		m_aActiveProbeIds.Clear();
		foreach (string activeId : activeIds)
		{
			OnProbeStop(activeId, DC_ProbeRunner.REASON_DELETE);
		}

		super.OnDelete(owner);
	}
}

//------------------------------------------------------------------------------------------------
//! What a running probe gets from the runner. One per started probe id; kept by the runner for the whole run.
class DC_ProbeContext
{
	protected string m_sProbeId;
	protected string m_sPlayDir;
	protected DC_ProbeRunner m_Runner;
	protected ref array<ref DC_ProbeCsv> m_aCsvs = {};	//!< every CSV opened through OpenCsv, for the footer
	protected ref array<string> m_aCsvNames = {};
	protected string m_sReportedFailure;				//!< first failure of a file not opened through OpenCsv

	//------------------------------------------------------------------------------------------------
	void DC_ProbeContext(string probeId, string playDir, DC_ProbeRunner runner)
	{
		m_sProbeId = probeId;
		m_sPlayDir = playDir;
		m_Runner = runner;
	}

	//------------------------------------------------------------------------------------------------
	string GetProbeId()
	{
		return m_sProbeId;
	}

	//------------------------------------------------------------------------------------------------
	//! $profile:DC_Probes/<probe>/<launch_id>/play_NN
	string GetPlayDir()
	{
		return m_sPlayDir;
	}

	//------------------------------------------------------------------------------------------------
	DC_ProbeRunner GetRunner()
	{
		return m_Runner;
	}

	//------------------------------------------------------------------------------------------------
	DC_LaunchInfo GetLaunchInfo()
	{
		if (!m_Runner)
			return null;

		return m_Runner.GetLaunchInfo();
	}

	//------------------------------------------------------------------------------------------------
	//! null when the layer has no entity named m_sEgoName
	IEntity GetEgo()
	{
		if (!m_Runner)
			return null;

		return m_Runner.GetEgo();
	}

	//------------------------------------------------------------------------------------------------
	string GetMapId()
	{
		if (!m_Runner)
			return string.Empty;

		return m_Runner.GetMapId();
	}

	//------------------------------------------------------------------------------------------------
	//! <play dir>/<name>.csv with its header row, registered for the footer; null on failure
	DC_ProbeCsv OpenCsv(string name, notnull array<string> columns)
	{
		DC_ProbeCsv csv = new DC_ProbeCsv();
		bool opened = csv.Open(m_sPlayDir + "/" + name + ".csv", columns);
		m_aCsvs.Insert(csv);
		m_aCsvNames.Insert(name);
		if (!opened)
		{
			Error("cannot open " + name + ".csv: " + csv.GetFailure());
			return null;
		}

		return csv;
	}

	//------------------------------------------------------------------------------------------------
	//! <play dir>/<name> written in one go (key=value lines); false on failure
	bool WriteTextFile(string name, notnull array<string> lines)
	{
		string failure;
		if (DC_ProbeCsv.WriteLines(m_sPlayDir + "/" + name, lines, false, failure))
			return true;

		Error("cannot write " + name + ": " + failure);
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! Appends key=value lines to <play dir>/run_header.txt (for values a probe only knows in or after OnProbeStart,
	//! for example rig.*, fiducial.*, area.*, vehicle.*). Readers take the last value of a repeated key. false on failure
	bool AppendHeaderLines(notnull array<string> lines)
	{
		string failure;
		if (DC_ProbeCsv.WriteLines(m_sPlayDir + "/" + DC_ProbeRunner.RUN_HEADER, lines, true, failure))
			return true;

		Error("cannot append to " + DC_ProbeRunner.RUN_HEADER + ": " + failure);
		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! <play dir>/<name>/ created (FileIO.MakeDirectory); its path, or empty on failure
	string MakeSubdir(string name)
	{
		string path = m_sPlayDir + "/" + name;
		if (!FileIO.MakeDirectory(path))
		{
			Error("cannot create " + path);
			return string.Empty;
		}

		return path;
	}

	//------------------------------------------------------------------------------------------------
	//! A probe reports a failed write of a file it writes itself (not through OpenCsv, e.g. the e19kill chunks); the
	//! runner then stops the probe with reason write_fail like any failed CSV. The first report is kept.
	void ReportWriteFailure(string failure)
	{
		if (!m_sReportedFailure.IsEmpty())
			return;

		m_sReportedFailure = failure;
		if (m_sReportedFailure.IsEmpty())
			m_sReportedFailure = "write failure reported by the probe";
	}

	//------------------------------------------------------------------------------------------------
	//! true once any CSV opened through this context failed a write, or the probe reported a failed write
	bool HasWriteFailure()
	{
		if (!m_sReportedFailure.IsEmpty())
			return true;

		foreach (DC_ProbeCsv csv : m_aCsvs)
		{
			if (csv && csv.HasFailed())
				return true;
		}

		return false;
	}

	//------------------------------------------------------------------------------------------------
	//! The first failure message of the first failed CSV (else the reported failure), empty when none failed
	string GetWriteFailure()
	{
		foreach (DC_ProbeCsv csv : m_aCsvs)
		{
			if (csv && csv.HasFailed())
				return csv.GetFailure();
		}

		return m_sReportedFailure;
	}

	//------------------------------------------------------------------------------------------------
	//! "[DC_Probe <id>] " + message, LogLevel.NORMAL
	void Log(string message)
	{
		Print("[DC_Probe " + m_sProbeId + "] " + message, LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	void Warn(string message)
	{
		Print("[DC_Probe " + m_sProbeId + "] WARNING: " + message, LogLevel.WARNING);
	}

	//------------------------------------------------------------------------------------------------
	void Error(string message)
	{
		Print("[DC_Probe " + m_sProbeId + "] ERROR: " + message, LogLevel.ERROR);
	}

	//------------------------------------------------------------------------------------------------
	//! Runner side: closes every CSV (idempotent)
	void CloseAll()
	{
		foreach (DC_ProbeCsv csv : m_aCsvs)
		{
			if (csv)
				csv.Close();
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Runner side: files=<n>, then file.<name>.rows / .bytes / .failed per CSV
	void AppendFooterLines(notnull array<string> lines)
	{
		int count = m_aCsvs.Count();
		lines.Insert("files=" + count.ToString());
		for (int i = 0; i < count; i++)
		{
			DC_ProbeCsv csv = m_aCsvs[i];
			string prefix = "file." + m_aCsvNames[i];
			int rows = csv.GetRowCount();
			int bytes = csv.GetByteCount();
			lines.Insert(prefix + ".rows=" + rows.ToString());
			lines.Insert(prefix + ".bytes=" + bytes.ToString());
			lines.Insert(prefix + ".failed=" + DC_LaunchGuard.BoolText(csv.HasFailed()));
		}
	}
}

//------------------------------------------------------------------------------------------------
//! Runner bookkeeping of one started probe id
class DC_ProbeRunState
{
	string m_sProbeId;
	DC_ProbeComponent m_Component;
	ref DC_ProbeContext m_Context;
	bool m_bConcurrent;
	bool m_bActive;				//!< OnProbeStart returned true and the probe has not been stopped
	bool m_bFinished;			//!< run_footer.txt written (or attempted)
	float m_fTimeoutS;
	float m_fStartWorldMs;
	int m_iStartStep;
	int m_iStartFrame;
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "T0-ENG probe runner: launch guard, probe plan, physics-step and frame dispatch, run headers and footers")]
class DC_ProbeRunnerClass : SCR_BaseGameModeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeRunner : SCR_BaseGameModeComponent
{
	// === PROBES ===
	[Attribute("e10,e04,e20,e19,e08a,e01,e02a,e05,e06a", UIWidgets.EditBox, "Probe ids to run, in this order (comma separated). e10 runs alongside the others; e19kill is never in the default plan. -dcProbePlan <ids> on the command line overrides this when the command line is visible", category: "Probes")]
	protected string m_sPlan;

	[Attribute("DC_Ego", UIWidgets.EditBox, "Name of the ego vehicle in the probe layer; probes that need it refuse to start without it", category: "Probes")]
	protected string m_sEgoName;

	[Attribute("arland", UIWidgets.EditBox, "Map id written to every run header (and used in E-08a file names)", category: "Probes")]
	protected string m_sMapId;

	[Attribute("Arland", UIWidgets.EditBox, "Terrain name written to every run header", category: "Probes")]
	protected string m_sTerrain;

	[Attribute("E:/DriveDiT", UIWidgets.EditBox, "Launch guard: -profile and $profile must lie inside this folder (case-insensitive, either slash). The probe guard has no off switch: empty refuses", category: "Probes")]
	protected string m_sRequiredProfileRoot;

	[Attribute("0", UIWidgets.CheckBox, "Launch guard: Workbench-only escape for a Play that cannot see the command line (E-11). It still needs the newest -wb- launch record in $profile:dc_launch, copied after the deploy by wb_R1.bat, and a $profile inside the root; escape runs are marked launch.mode=wb_escape", category: "Probes")]
	protected bool m_bAllowWorkbenchWithoutLaunchId;

	[Attribute("1", UIWidgets.CheckBox, "Launch guard escape only: the newest -wb- record must be this Workbench start's own record (written at most 300 s before the process start, estimated as GetUnixTime - GetTickCount / 1000); there is no fall-back to an older record. Off = the newest record younger than the deploy (launch.escape_binding=newest_after_deploy): only if E-11 shows that GetTickCount does not count from the process start", category: "Probes")]
	protected bool m_bEscapeBindToProcessStart;

	[Attribute("3", UIWidgets.Slider, "Warm-up before the first exclusive probe [world s]; the physics-step source is decided at its end", "0 60 0.5", category: "Probes")]
	protected float m_fWarmupS;

	[Attribute("1", UIWidgets.Slider, "Gap between two exclusive probes [world s]", "0 60 0.5", category: "Probes")]
	protected float m_fGapS;

	[Attribute("10000", UIWidgets.Slider, "Start this long after OnPostInit if OnGameModeStart never arrived [ms], 0 = no fallback", "0 60000 500", category: "Probes")]
	protected int m_iFallbackStartMs;

	// === CONSTANTS ===
	static const string PROBE_ROOT = "$profile:DC_Probes";
	static const string RUN_HEADER = "run_header.txt";
	static const string RUN_FOOTER = "run_footer.txt";
	static const string RUN_SCHEMA = "dc-probe-run/1";
	static const string CLI_PLAN = "dcProbePlan";
	static const int MAX_PLAYS = 99;

	static const string STEP_SIMULATE = "simulate";
	static const string STEP_FIXEDFRAME = "fixedframe";
	static const string STEP_NONE = "none";

	static const string REASON_DONE = "done";
	static const string REASON_TIMEOUT = "timeout";
	static const string REASON_WRITE_FAIL = "write_fail";
	static const string REASON_ERROR = "error";
	static const string REASON_PLAY_END = "play_end";
	static const string REASON_DELETE = "delete";
	static const string REASON_START_FAILED = "start_failed";

	static const string STATUS_COMPLETE = "complete";
	static const string STATUS_STOPPED = "stopped";
	static const string STATUS_FAILED = "failed";

	// === STATE ===
	protected bool m_bStarted;			//!< StartRun ran (any machine), never cleared
	protected bool m_bRunning;			//!< events are dispatched and the plan is sequenced
	protected bool m_bMaskSet;			//!< SetEventMask ran and ClearEventMask has not yet
	protected float m_fFallbackArmedMs;	//!< world time when FallbackStart was queued in OnPostInit
	protected ref DC_LaunchInfo m_LaunchInfo;
	protected IEntity m_Ego;
	protected ref array<string> m_aPlanIds = {};				//!< plan as run (ids with a component, no duplicates)
	protected ref array<string> m_aExclusiveIds = {};
	protected ref array<DC_ProbeComponent> m_aExclusiveComponents = {};
	protected ref array<string> m_aConcurrentIds = {};
	protected ref array<DC_ProbeComponent> m_aConcurrentComponents = {};
	protected ref array<ref DC_ProbeRunState> m_aRuns = {};		//!< every started probe, kept for the whole run
	protected ref array<DC_ProbeRunState> m_aConcurrentRuns = {};
	protected DC_ProbeRunState m_CurrentRun;					//!< running or just-stopped exclusive probe
	protected int m_iNextExclusive;
	protected float m_fNextStartAtMs;
	protected float m_fRunStartWorldMs;
	protected string m_sStepSource;		//!< "" until the warm-up ends, then simulate | fixedframe | none

	protected int m_iPhysicsStep;
	protected int m_iSimulateCount;
	protected int m_iPostSimulateCount;
	protected int m_iFixedFrameCount;
	protected int m_iPostFixedFrameCount;
	protected int m_iFrameCount;

	//------------------------------------------------------------------------------------------------
	override void OnPostInit(IEntity owner)
	{
		super.OnPostInit(owner);

		// OnPostInit also runs when the World Editor loads the layer: do nothing in edit mode.
		if (SCR_Global.IsEditMode(owner) || SCR_Global.IsEditMode())
			return;

		Print(string.Format("[DC_Probe] Runner initialized on '%1' (server: %2), waiting for OnGameModeStart", owner.GetName(), Replication.IsServer()), LogLevel.NORMAL);

		if (m_iFallbackStartMs > 0)
		{
			BaseWorld world = owner.GetWorld();
			if (world)
				m_fFallbackArmedMs = world.GetWorldTime();

			GetGame().GetCallqueue().CallLater(FallbackStart, m_iFallbackStartMs, false);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Called from SCR_BaseGameMode.OnGameStateChanged() when the state becomes GAME (on every machine).
	override void OnGameModeStart()
	{
		super.OnGameModeStart();
		StartRun("OnGameModeStart");
	}

	//------------------------------------------------------------------------------------------------
	protected void FallbackStart()
	{
		if (m_bStarted)
			return;

		// Same rule as the DD-1 director: fall back only once m_iFallbackStartMs of world time have passed since
		// OnPostInit (the first frame after a long world load can carry a large timeslice); until then re-queue.
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

		Print(string.Format("[DC_Probe] WARNING: OnGameModeStart not received within %1 ms", m_iFallbackStartMs), LogLevel.WARNING);
		StartRun("FallbackStart");
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent: only the first call does anything.
	protected void StartRun(string reason)
	{
		if (m_bStarted)
			return;

		if (!GetGame().InPlayMode())
			return;

		m_bStarted = true;
		GetGame().GetCallqueue().Remove(FallbackStart);

		if (!Replication.IsServer())
		{
			Print("[DC_Probe] Runner: not the server, nothing to do on this machine", LogLevel.NORMAL);
			return;
		}

		// The guard never prints, so the E-11 line comes first. It shows $profile: never paste it into a PR or report.
		m_LaunchInfo = DC_LaunchGuard.Evaluate(m_sRequiredProfileRoot, m_bAllowWorkbenchWithoutLaunchId, m_bEscapeBindToProcessStart);
		string profileAbs = m_LaunchInfo.m_sProfileAbs;
		if (profileAbs.IsEmpty())
			profileAbs = "n/a";

		// uptime_s = GetTickCount / 1000: shows whether the tick count runs from the process start or the Play start
		Print(string.Format("[DC_Probe] E-11: -profile visible=%1 -dcLaunchId visible=%2 $profile=%3 exe=%4 uptime_s=%5", DC_LaunchGuard.BoolText(m_LaunchInfo.m_bCliProfile), DC_LaunchGuard.BoolText(m_LaunchInfo.m_bCliLaunchId), profileAbs, GetEngineExe(), m_LaunchInfo.m_iUptimeS), LogLevel.NORMAL);

		if (!m_LaunchInfo.m_bAllowed)
		{
			Print("[DC_Probe] ERROR: launch guard refused: " + m_LaunchInfo.m_sReason + " - no probe runs, nothing written", LogLevel.ERROR);
			return;
		}

		string guardLine = string.Format("[DC_Probe] launch guard: mode=%1 id=%2 pin=%3", m_LaunchInfo.m_sMode, m_LaunchInfo.m_sLaunchId, m_LaunchInfo.m_sPinId);
		if (m_LaunchInfo.m_sMode == DC_LaunchGuard.MODE_ESCAPE)
			Print(guardLine + " (Workbench escape: the command line is not visible; joined the newest launch record, binding " + m_LaunchInfo.m_sEscapeBinding + ")", LogLevel.WARNING);
		else
			Print(guardLine, LogLevel.NORMAL);

		BaseWorld world = GetGame().GetWorld();
		m_Ego = world.FindEntityByName(m_sEgoName);
		if (!m_Ego)
			Print("[DC_Probe] WARNING: no entity named '" + m_sEgoName + "' in the world - probes that need the ego will refuse to start", LogLevel.WARNING);

		BuildPlan();

		m_fRunStartWorldMs = world.GetWorldTime();
		m_fNextStartAtMs = m_fRunStartWorldMs;
		m_bRunning = true;
		SetEventMask(GetOwner(), GetDispatchEventMask());
		m_bMaskSet = true;

		Print(string.Format("[DC_Probe] run started (%1): plan=%2, warm-up %3 s, gap %4 s", reason, JoinIds(m_aPlanIds), m_fWarmupS.ToString(-1, 1), m_fGapS.ToString(-1, 1)), LogLevel.NORMAL);

		for (int i = 0, count = m_aConcurrentIds.Count(); i < count; i++)
		{
			DC_ProbeRunState state = StartProbe(m_aConcurrentIds[i], m_aConcurrentComponents[i], true);
			if (state && state.m_bActive)
				m_aConcurrentRuns.Insert(state);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! -dcProbePlan when visible, else m_sPlan; split on ',', trimmed. Ids without a component and duplicates are
	//! skipped with a WARNING. Fills m_aPlanIds and the exclusive and concurrent lists.
	protected void BuildPlan()
	{
		string planText = m_sPlan;
		string planSource = "m_sPlan";
		string cliPlan;
		if (System.GetCLIParam(CLI_PLAN, cliPlan) && !cliPlan.Trim().IsEmpty())
		{
			planText = cliPlan;
			planSource = "-dcProbePlan";
		}

		array<Managed> found = {};
		GetOwner().FindComponents(DC_ProbeComponent, found);

		array<string> tokens = {};
		planText.Split(",", tokens, true);
		foreach (string token : tokens)
		{
			string probeId = token.Trim();
			if (probeId.IsEmpty())
				continue;

			if (m_aPlanIds.Find(probeId) >= 0)
			{
				Print("[DC_Probe] WARNING: plan id '" + probeId + "' is listed twice - the repeat is skipped", LogLevel.WARNING);
				continue;
			}

			DC_ProbeComponent component = FindProbeComponent(found, probeId);
			if (!component)
			{
				Print("[DC_Probe] WARNING: plan id '" + probeId + "' has no probe component on this game mode - skipped", LogLevel.WARNING);
				continue;
			}

			m_aPlanIds.Insert(probeId);
			if (component.IsConcurrent(probeId))
			{
				m_aConcurrentIds.Insert(probeId);
				m_aConcurrentComponents.Insert(component);
			}
			else
			{
				m_aExclusiveIds.Insert(probeId);
				m_aExclusiveComponents.Insert(component);
			}
		}

		Print(string.Format("[DC_Probe] plan from %1: '%2' -> %3 probe(s) with a component: %4 (%5 probe component(s) on the game mode)", planSource, planText, m_aPlanIds.Count(), JoinIds(m_aPlanIds), found.Count()), LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! \return the first probe component that implements probeId, null if none does
	protected DC_ProbeComponent FindProbeComponent(notnull array<Managed> components, string probeId)
	{
		DC_ProbeComponent result;
		array<string> ids = {};
		foreach (Managed managed : components)
		{
			DC_ProbeComponent component = DC_ProbeComponent.Cast(managed);
			if (!component)
				continue;

			ids.Clear();
			component.GetProbeIds(ids);
			if (ids.Find(probeId) < 0)
				continue;

			if (!result)
				result = component;
			else
				Print("[DC_Probe] WARNING: probe id '" + probeId + "' is implemented by more than one component - using the first", LogLevel.WARNING);
		}

		return result;
	}

	//------------------------------------------------------------------------------------------------
	//! Called on every machine when the game mode ends (SCR_BaseGameMode.OnGameStateChanged, POSTGAME).
	override void OnGameModeEnd(SCR_GameModeEndData data)
	{
		super.OnGameModeEnd(data);
		StopRun(REASON_PLAY_END);
	}

	//------------------------------------------------------------------------------------------------
	//! Called on every machine when the world ends, before the entities are deleted: probes restore what they changed.
	override void OnGameEnd()
	{
		super.OnGameEnd();
		StopRun(REASON_PLAY_END);
	}

	//------------------------------------------------------------------------------------------------
	override void OnDelete(IEntity owner)
	{
		// The probe components stop themselves in their own OnDelete; do not touch another component during deletion.
		StopRun(REASON_DELETE);
		super.OnDelete(owner);
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent. play_end: the running probes get OnProbeStop(play_end). delete: no probe component is called; the
	//! contexts' files are closed and the footers written (status stopped, reason delete). Both remove FallbackStart
	//! and clear the event mask.
	protected void StopRun(string reason)
	{
		ArmaReforgerScripted game = GetGame();
		if (game)
		{
			ScriptCallQueue queue = game.GetCallqueue();
			if (queue)
				queue.Remove(FallbackStart);
		}

		bool wasRunning = m_bRunning;
		m_bRunning = false;

		if (reason == REASON_DELETE)
		{
			foreach (DC_ProbeRunState state : m_aRuns)
			{
				FinishProbeForDelete(state);
			}
		}
		else
		{
			if (m_CurrentRun)
				StopProbe(m_CurrentRun, reason);

			foreach (DC_ProbeRunState concurrent : m_aConcurrentRuns)
			{
				StopProbe(concurrent, reason);
			}
		}

		ClearDispatchMask();
		if (wasRunning)
			Print("[DC_Probe] WARNING: run stopped (" + reason + ") before the plan completed", LogLevel.WARNING);
	}

	//------------------------------------------------------------------------------------------------
	protected void ClearDispatchMask()
	{
		if (!m_bMaskSet)
			return;

		m_bMaskSet = false;
		IEntity owner = GetOwner();
		if (owner)
			ClearEventMask(owner, GetDispatchEventMask());
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
		if (!m_bRunning)
			return;

		if (m_sStepSource == STEP_SIMULATE || m_sStepSource.IsEmpty())
		{
			m_iPhysicsStep++;
			DispatchSimulate(timeSlice);
		}
	}

	//------------------------------------------------------------------------------------------------
	override void EOnPostSimulate(IEntity owner, float timeSlice)
	{
		m_iPostSimulateCount++;
		if (!m_bRunning)
			return;

		if (m_sStepSource == STEP_SIMULATE || m_sStepSource.IsEmpty())
			DispatchPostSimulate(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void EOnFixedFrame(IEntity owner, float timeSlice)
	{
		m_iFixedFrameCount++;
		if (!m_bRunning)
			return;

		if (m_sStepSource == STEP_FIXEDFRAME)
		{
			m_iPhysicsStep++;
			DispatchSimulate(timeSlice);
		}
	}

	//------------------------------------------------------------------------------------------------
	override void EOnPostFixedFrame(IEntity owner, float timeSlice)
	{
		m_iPostFixedFrameCount++;
		if (!m_bRunning)
			return;

		if (m_sStepSource == STEP_FIXEDFRAME)
			DispatchPostSimulate(timeSlice);
	}

	//------------------------------------------------------------------------------------------------
	override void EOnFrame(IEntity owner, float timeSlice)
	{
		m_iFrameCount++;
		if (!m_bRunning)
			return;

		int renderFrame = GetRenderFrame();
		if (m_CurrentRun && m_CurrentRun.m_bActive && m_CurrentRun.m_Component)
			m_CurrentRun.m_Component.OnProbeFrame(m_CurrentRun.m_sProbeId, timeSlice, renderFrame);

		foreach (DC_ProbeRunState state : m_aConcurrentRuns)
		{
			if (state.m_bActive && state.m_Component)
				state.m_Component.OnProbeFrame(state.m_sProbeId, timeSlice, renderFrame);
		}

		Sequence();
	}

	//------------------------------------------------------------------------------------------------
	//! OnProbeSimulate on the running exclusive probe, then on the concurrent ones
	protected void DispatchSimulate(float timeSlice)
	{
		if (m_CurrentRun && m_CurrentRun.m_bActive && m_CurrentRun.m_Component)
			m_CurrentRun.m_Component.OnProbeSimulate(m_CurrentRun.m_sProbeId, timeSlice, m_iPhysicsStep);

		foreach (DC_ProbeRunState state : m_aConcurrentRuns)
		{
			if (state.m_bActive && state.m_Component)
				state.m_Component.OnProbeSimulate(state.m_sProbeId, timeSlice, m_iPhysicsStep);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! OnProbePostSimulate with the number of the step's pre call (or the last number, for a post without a pre)
	protected void DispatchPostSimulate(float timeSlice)
	{
		if (m_CurrentRun && m_CurrentRun.m_bActive && m_CurrentRun.m_Component)
			m_CurrentRun.m_Component.OnProbePostSimulate(m_CurrentRun.m_sProbeId, timeSlice, m_iPhysicsStep);

		foreach (DC_ProbeRunState state : m_aConcurrentRuns)
		{
			if (state.m_bActive && state.m_Component)
				state.m_Component.OnProbePostSimulate(state.m_sProbeId, timeSlice, m_iPhysicsStep);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Once per frame after the dispatch: warm-up decision, stop checks, next exclusive probe, end of the plan
	protected void Sequence()
	{
		float now = GetWorldTimeMs();
		if (m_sStepSource.IsEmpty() && now - m_fRunStartWorldMs >= Math.Max(m_fWarmupS, 0) * 1000)
			DecideStepSource(now);

		// Concurrent probes run until the plan ends; they only stop early on a failed write or their timeout.
		foreach (DC_ProbeRunState concurrent : m_aConcurrentRuns)
		{
			if (!concurrent.m_bActive)
				continue;

			if (concurrent.m_Context.HasWriteFailure())
				StopProbe(concurrent, REASON_WRITE_FAIL);
			else if (now - concurrent.m_fStartWorldMs >= concurrent.m_fTimeoutS * 1000)
				StopProbe(concurrent, REASON_TIMEOUT);
		}

		if (m_CurrentRun)
		{
			if (m_CurrentRun.m_bActive)
			{
				if (m_CurrentRun.m_Context.HasWriteFailure())
					StopProbe(m_CurrentRun, REASON_WRITE_FAIL);
				else if (!m_CurrentRun.m_Component || m_CurrentRun.m_Component.IsProbeDone(m_CurrentRun.m_sProbeId))
					StopProbe(m_CurrentRun, REASON_DONE);
				else if (now - m_CurrentRun.m_fStartWorldMs >= m_CurrentRun.m_fTimeoutS * 1000)
					StopProbe(m_CurrentRun, REASON_TIMEOUT);
			}

			if (m_CurrentRun.m_bActive)
				return;

			m_CurrentRun = null;
			m_fNextStartAtMs = now + Math.Max(m_fGapS, 0) * 1000;
			return;
		}

		// Exclusive probes start only after the warm-up, m_fGapS apart
		if (m_sStepSource.IsEmpty() || now < m_fNextStartAtMs)
			return;

		if (m_iNextExclusive < m_aExclusiveIds.Count())
		{
			string probeId = m_aExclusiveIds[m_iNextExclusive];
			DC_ProbeComponent component = m_aExclusiveComponents[m_iNextExclusive];
			m_iNextExclusive++;

			DC_ProbeRunState state = StartProbe(probeId, component, false);
			if (state && state.m_bActive)
				m_CurrentRun = state;
			else
				m_fNextStartAtMs = now + Math.Max(m_fGapS, 0) * 1000;

			return;
		}

		FinishPlan();
	}

	//------------------------------------------------------------------------------------------------
	//! End of the warm-up: simulate if any EOnSimulate arrived, else fixedframe if any EOnFixedFrame arrived, else none
	protected void DecideStepSource(float now)
	{
		float warmupS = (now - m_fRunStartWorldMs) / 1000;
		string counts = string.Format("EOnSimulate %1, EOnPostSimulate %2, EOnFixedFrame %3, EOnPostFixedFrame %4, EOnFrame %5 in %6 s", m_iSimulateCount, m_iPostSimulateCount, m_iFixedFrameCount, m_iPostFixedFrameCount, m_iFrameCount, warmupS.ToString(-1, 2));
		if (m_iSimulateCount > 0)
		{
			m_sStepSource = STEP_SIMULATE;
			Print("[DC_Probe] step source: simulate (" + counts + ")", LogLevel.NORMAL);
		}
		else if (m_iFixedFrameCount > 0)
		{
			m_sStepSource = STEP_FIXEDFRAME;
			Print("[DC_Probe] WARNING: step source: fixedframe - no EOnSimulate on the game mode entity, physics steps come from EOnFixedFrame (" + counts + ")", LogLevel.WARNING);
		}
		else
		{
			m_sStepSource = STEP_NONE;
			Print("[DC_Probe] ERROR: step source: none - neither EOnSimulate nor EOnFixedFrame arrived, probes that need physics steps will refuse to start (" + counts + ")", LogLevel.ERROR);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Allocates the play dir, writes run_header.txt, logs START and calls OnProbeStart.
	//! \return the run state (m_bActive when the probe runs), or null when no play dir could be created
	protected DC_ProbeRunState StartProbe(string probeId, DC_ProbeComponent component, bool concurrent)
	{
		string playDir = AllocatePlayDir(probeId);
		if (playDir.IsEmpty())
		{
			Print("[DC_Probe " + probeId + "] ERROR: no play directory could be created under " + GetProbeBaseDir(probeId) + " - probe skipped", LogLevel.ERROR);
			return null;
		}

		DC_ProbeRunState state = new DC_ProbeRunState();
		state.m_sProbeId = probeId;
		state.m_Component = component;
		state.m_bConcurrent = concurrent;
		state.m_Context = new DC_ProbeContext(probeId, playDir, this);
		state.m_fTimeoutS = component.GetProbeTimeoutS(probeId);
		state.m_fStartWorldMs = GetWorldTimeMs();
		state.m_iStartStep = m_iPhysicsStep;
		state.m_iStartFrame = m_iFrameCount;
		m_aRuns.Insert(state);

		array<string> header = {};
		BuildRunHeader(state, header);
		if (!state.m_Context.WriteTextFile(RUN_HEADER, header))
		{
			FinishProbe(state, STATUS_FAILED, REASON_START_FAILED, "run_header.txt could not be written");
			return state;
		}

		Print("[DC_Probe " + probeId + "] START play_dir=" + playDir, LogLevel.NORMAL);

		string refusal;
		if (component.NeedsEgo(probeId) && !m_Ego)
			refusal = "no ego: the world has no entity named '" + m_sEgoName + "'";
		else if (component.NeedsSteps(probeId) && m_sStepSource == STEP_NONE)
			refusal = "no physics step source: neither EOnSimulate nor EOnFixedFrame arrived in the warm-up";

		if (!refusal.IsEmpty())
		{
			Print("[DC_Probe " + probeId + "] ERROR: not started: " + refusal, LogLevel.ERROR);
			FinishProbe(state, STATUS_FAILED, REASON_START_FAILED, refusal);
			return state;
		}

		component.MarkProbeActive(probeId, true);
		if (!component.OnProbeStart(probeId, state.m_Context))
		{
			// Let the probe undo a partial start (camera, settings, markers)
			component.OnProbeStop(probeId, REASON_ERROR);
			component.MarkProbeActive(probeId, false);
			Print("[DC_Probe " + probeId + "] ERROR: OnProbeStart failed - probe not started", LogLevel.ERROR);
			FinishProbe(state, STATUS_FAILED, REASON_START_FAILED, "OnProbeStart returned false " + state.m_Context.GetWriteFailure());
			return state;
		}

		state.m_bActive = true;
		return state;
	}

	//------------------------------------------------------------------------------------------------
	//! Normal stop: OnProbeStop(reason), then the footer. Idempotent.
	protected void StopProbe(notnull DC_ProbeRunState state, string reason)
	{
		if (!state.m_bActive)
			return;

		state.m_bActive = false;
		if (state.m_Component)
		{
			state.m_Component.OnProbeStop(state.m_sProbeId, reason);
			state.m_Component.MarkProbeActive(state.m_sProbeId, false);
		}

		// A write that failed inside OnProbeStop (the last rows) fails the probe as well: the footer must not say
		// complete next to file.<name>.failed=1
		string stopNote;
		if (reason != REASON_WRITE_FAIL && state.m_Context.HasWriteFailure())
		{
			stopNote = "write failed while stopping (reason " + reason + "): ";
			reason = REASON_WRITE_FAIL;
		}

		string status = STATUS_STOPPED;
		if (reason == REASON_DONE)
			status = STATUS_COMPLETE;
		else if (reason == REASON_WRITE_FAIL || reason == REASON_ERROR)
			status = STATUS_FAILED;

		string note = stopNote + state.m_Context.GetWriteFailure();
		if (reason == REASON_TIMEOUT)
			note = string.Format("timeout after %1 world s %2", state.m_fTimeoutS.ToString(-1, 1), note);

		if (reason == REASON_WRITE_FAIL)
			Print("[DC_Probe " + state.m_sProbeId + "] ERROR: write failure: " + note, LogLevel.ERROR);

		FinishProbe(state, status, reason, note);
	}

	//------------------------------------------------------------------------------------------------
	//! World teardown: closes the context's files and writes the footer; calls no probe component. Idempotent.
	protected void FinishProbeForDelete(notnull DC_ProbeRunState state)
	{
		if (state.m_bFinished)
			return;

		state.m_bActive = false;
		FinishProbe(state, STATUS_STOPPED, REASON_DELETE, "world teardown: the probe component stops itself in its own OnDelete");
	}

	//------------------------------------------------------------------------------------------------
	//! Closes the context's CSVs, writes run_footer.txt and logs END
	protected void FinishProbe(notnull DC_ProbeRunState state, string status, string reason, string note)
	{
		state.m_bActive = false;
		state.m_bFinished = true;
		state.m_Context.CloseAll();

		array<string> footer = {};
		BuildRunFooter(state, status, reason, note, footer);
		state.m_Context.WriteTextFile(RUN_FOOTER, footer);

		string line = "[DC_Probe " + state.m_sProbeId + "] END reason=" + reason + " status=" + status;
		if (status == STATUS_COMPLETE)
			Print(line, LogLevel.NORMAL);
		else if (status == STATUS_STOPPED)
			Print(line + " " + note, LogLevel.WARNING);
		else
			Print(line + " " + note, LogLevel.ERROR);
	}

	//------------------------------------------------------------------------------------------------
	//! Plan finished: the concurrent probes stop with reason done and the event mask is cleared
	protected void FinishPlan()
	{
		foreach (DC_ProbeRunState concurrent : m_aConcurrentRuns)
		{
			StopProbe(concurrent, REASON_DONE);
		}

		m_bRunning = false;
		ClearDispatchMask();
		Print(string.Format("[DC_Probe] plan complete: %1 probe(s) started (%2), %3 physics steps, %4 frames, step source %5", m_aRuns.Count(), JoinIds(m_aPlanIds), m_iPhysicsStep, m_iFrameCount, m_sStepSource), LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! $profile:DC_Probes/<probe>/<launch_id>
	protected string GetProbeBaseDir(string probeId)
	{
		return PROBE_ROOT + "/" + probeId + "/" + m_LaunchInfo.m_sLaunchId;
	}

	//------------------------------------------------------------------------------------------------
	//! play_NN, NN = 01..99: the first one for which neither the directory nor run_header.txt exists. Created here.
	//! \return its path, or empty when none is free or MakeDirectory fails
	protected string AllocatePlayDir(string probeId)
	{
		string base = GetProbeBaseDir(probeId);
		for (int n = 1; n <= MAX_PLAYS; n++)
		{
			string dir = base + "/play_" + n.ToString(2);
			if (FileIO.FileExists(dir) || FileIO.FileExists(dir + "/" + RUN_HEADER))
				continue;

			if (!FileIO.MakeDirectory(dir))
				return string.Empty;

			return dir;
		}

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	//! run_header.txt keys of PROBE_DESIGN.md 4.5, in that order
	protected void BuildRunHeader(notnull DC_ProbeRunState state, notnull array<string> lines)
	{
		string probeId = state.m_sProbeId;
		lines.Insert("schema=" + RUN_SCHEMA);
		lines.Insert("probe=" + probeId);
		lines.Insert("plan=" + JoinIds(m_aPlanIds));
		lines.Insert("play_dir=" + state.m_Context.GetPlayDir());
		m_LaunchInfo.AppendHeaderLines(lines);

		// Provenance of the deployed addon, generated by tools/deploy_addon.py
		int contentFileCount = DC_BuildInfo.CONTENT_FILE_COUNT;
		lines.Insert("addon.git_sha=" + DC_BuildInfo.GIT_SHA);
		lines.Insert("addon.git_branch=" + DC_BuildInfo.GIT_BRANCH);
		lines.Insert("addon.git_dirty=" + DC_LaunchGuard.BoolText(DC_BuildInfo.GIT_DIRTY));
		lines.Insert("addon.deploy_utc=" + DC_BuildInfo.DEPLOY_UTC);
		lines.Insert("addon.manifest_sha256=" + DC_BuildInfo.MANIFEST_SHA256);
		lines.Insert("addon.content_sha256=" + DC_BuildInfo.CONTENT_SHA256);
		lines.Insert("addon.content_file_count=" + contentFileCount.ToString());
		lines.Insert("addon.deploy_partial=" + DC_LaunchGuard.BoolText(DC_BuildInfo.DEPLOY_PARTIAL));

		ArmaReforgerScripted game = GetGame();
		BaseWorld world = game.GetWorld();
		float fixedTimeSlice = world.GetFixedTimeSlice();
		lines.Insert("engine.build_version=" + game.GetBuildVersion());
		lines.Insert("engine.build_time=" + game.GetBuildTime());
		lines.Insert("engine.fixed_time_slice_s=" + fixedTimeSlice.ToString(-1, 6));
		lines.Insert("engine.exe=" + GetEngineExe());
		lines.Insert("engine.gproj=" + GetGameProjectFile());

		lines.Insert("world.file=" + game.GetWorldFile());
		lines.Insert("world.map_id=" + m_sMapId);
		lines.Insert("world.terrain=" + m_sTerrain);

		lines.Insert("authority.is_server=" + DC_LaunchGuard.BoolText(Replication.IsServer()));
		lines.Insert("authority.rpl_running=" + DC_LaunchGuard.BoolText(Replication.IsRunning()));
		lines.Insert("authority.rpl_mode=" + typename.EnumToString(RplMode, RplSession.Mode()));

		int unixTime = System.GetUnixTime();
		int tickMs = System.GetTickCount();
		int renderFrame = GetRenderFrame();
		lines.Insert("clock.start_unix=" + unixTime.ToString());
		lines.Insert("clock.start_utc=" + FormatUtcNow());
		lines.Insert("clock.start_tick_ms=" + tickMs.ToString());
		lines.Insert("clock.start_world_ms=" + state.m_fStartWorldMs.ToString(-1, 3));
		lines.Insert("clock.start_render_frame=" + renderFrame.ToString());
		lines.Insert("clock.start_physics_step=" + state.m_iStartStep.ToString());
		lines.Insert("clock.step_source=" + m_sStepSource);

		string egoPrefab;
		if (m_Ego)
		{
			EntityPrefabData prefabData = m_Ego.GetPrefabData();
			if (prefabData)
				egoPrefab = prefabData.GetPrefabName();
		}

		lines.Insert("ego.name=" + m_sEgoName);
		lines.Insert("ego.found=" + DC_LaunchGuard.BoolText(m_Ego != null));
		lines.Insert("ego.prefab=" + egoPrefab);

		array<string> keyValues = {};
		if (state.m_Component)
			state.m_Component.GetProbeParams(probeId, keyValues);

		foreach (string keyValue : keyValues)
		{
			lines.Insert("params." + keyValue);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! run_footer.txt keys of PROBE_DESIGN.md 4.5
	protected void BuildRunFooter(notnull DC_ProbeRunState state, string status, string reason, string note, notnull array<string> lines)
	{
		int unixTime = System.GetUnixTime();
		int tickMs = System.GetTickCount();
		int renderFrame = GetRenderFrame();
		int physicsSteps = m_iPhysicsStep - state.m_iStartStep;
		int renderFrames = m_iFrameCount - state.m_iStartFrame;
		float worldMs = GetWorldTimeMs();

		lines.Insert("schema=" + RUN_SCHEMA);
		lines.Insert("status=" + status);
		lines.Insert("reason=" + reason);
		lines.Insert("end.unix=" + unixTime.ToString());
		lines.Insert("end.utc=" + FormatUtcNow());
		lines.Insert("end.tick_ms=" + tickMs.ToString());
		lines.Insert("end.world_ms=" + worldMs.ToString(-1, 3));
		lines.Insert("end.render_frame=" + renderFrame.ToString());
		lines.Insert("end.physics_step=" + m_iPhysicsStep.ToString());
		lines.Insert("physics_steps=" + physicsSteps.ToString());
		lines.Insert("render_frames=" + renderFrames.ToString());
		state.m_Context.AppendFooterLines(lines);
		lines.Insert("note=" + note.Trim());
	}

	//------------------------------------------------------------------------------------------------
	protected static string JoinIds(notnull array<string> ids)
	{
		string text;
		foreach (string probeId : ids)
		{
			if (!text.IsEmpty())
				text += ",";

			text += probeId;
		}

		return text;
	}

	//------------------------------------------------------------------------------------------------
	//! "wb" in Workbench, "game" otherwise
	static string GetEngineExe()
	{
#ifdef WORKBENCH
		return "wb";
#else
		return "game";
#endif
	}

	//------------------------------------------------------------------------------------------------
	//! Workbench.GetCurrentGameProjectFile() in Workbench, empty otherwise
	static string GetGameProjectFile()
	{
#ifdef WORKBENCH
		return Workbench.GetCurrentGameProjectFile();
#else
		return string.Empty;
#endif
	}

	//------------------------------------------------------------------------------------------------
	//! Dispatched steps since StartRun (the step being or last simulated); 0 before the first
	int GetPhysicsStep()
	{
		return m_iPhysicsStep;
	}

	//------------------------------------------------------------------------------------------------
	//! BaseWorld.GetFrameNumber()
	int GetRenderFrame()
	{
		BaseWorld world = GetProbeWorld();
		if (!world)
			return -1;

		return world.GetFrameNumber();
	}

	//------------------------------------------------------------------------------------------------
	//! BaseWorld.GetWorldTime()
	float GetWorldTimeMs()
	{
		BaseWorld world = GetProbeWorld();
		if (!world)
			return 0;

		return world.GetWorldTime();
	}

	//------------------------------------------------------------------------------------------------
	//! The game's world, null during teardown
	protected BaseWorld GetProbeWorld()
	{
		ArmaReforgerScripted game = GetGame();
		if (!game)
			return null;

		return game.GetWorld();
	}

	//------------------------------------------------------------------------------------------------
	//! World ms since StartRun
	float GetRunWorldMs()
	{
		return GetWorldTimeMs() - m_fRunStartWorldMs;
	}

	//------------------------------------------------------------------------------------------------
	//! System.GetTickCount()
	int GetTickMs()
	{
		return System.GetTickCount();
	}

	//------------------------------------------------------------------------------------------------
	//! ChimeraWorld.IsGameTimePaused()
	bool IsPaused()
	{
		ChimeraWorld world = ChimeraWorld.CastFrom(GetProbeWorld());
		if (!world)
			return false;

		return world.IsGameTimePaused();
	}

	//------------------------------------------------------------------------------------------------
	//! "simulate", "fixedframe", "none" or "" (undecided, warm-up)
	string GetStepSource()
	{
		return m_sStepSource;
	}

	//------------------------------------------------------------------------------------------------
	//! EOnSimulate calls since StartRun
	int GetSimulateCount()
	{
		return m_iSimulateCount;
	}

	//------------------------------------------------------------------------------------------------
	int GetPostSimulateCount()
	{
		return m_iPostSimulateCount;
	}

	//------------------------------------------------------------------------------------------------
	int GetFixedFrameCount()
	{
		return m_iFixedFrameCount;
	}

	//------------------------------------------------------------------------------------------------
	int GetPostFixedFrameCount()
	{
		return m_iPostFixedFrameCount;
	}

	//------------------------------------------------------------------------------------------------
	//! EOnFrame calls since StartRun
	int GetFrameCount()
	{
		return m_iFrameCount;
	}

	//------------------------------------------------------------------------------------------------
	IEntity GetEgo()
	{
		return m_Ego;
	}

	//------------------------------------------------------------------------------------------------
	DC_LaunchInfo GetLaunchInfo()
	{
		return m_LaunchInfo;
	}

	//------------------------------------------------------------------------------------------------
	string GetMapId()
	{
		return m_sMapId;
	}

	//------------------------------------------------------------------------------------------------
	string GetTerrain()
	{
		return m_sTerrain;
	}

	//------------------------------------------------------------------------------------------------
	//! "YYYY-MM-DDTHH:MM:SSZ" from System.GetYearMonthDayUTC / GetHourMinuteSecondUTC. The date is read on both
	//! sides of the time, so a read across midnight takes the time again with the later date.
	static string FormatUtcNow()
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

		return string.Format("%1-%2-%3T%4:%5:%6Z", year.ToString(4), month.ToString(2), day.ToString(2), hour.ToString(2), minute.ToString(2), second.ToString(2));
	}
}
