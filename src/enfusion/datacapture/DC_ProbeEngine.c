// ============================================================================
// DC_ProbeEngine - T0-ENG engine-basics probes E-04, E-10, E-19 and E-20 (DD-4-probe)
// ============================================================================
//
// One component on GameMode_DCProbes next to DC_ProbeRunner, which drives it (DC_ProbeComponent). Probe ids:
//   e10      concurrent, runs for the whole plan. Counts every event the runner receives, per render frame
//            (e10_frames.csv) and per physics step (e10_steps.csv), and writes a telemetry row every k-th step
//            (e10_telemetry.csv), k = max(1, round(m_fE10TelemetryPeriodS / GetFixedTimeSlice())). r2 4.4: the
//            EOnPostSimulate count per world-second is 1/dt +-1 and the telemetry comes every k steps, 0 missed.
//            Step and telemetry rows also carry world_ts_ms = BaseWorld.GetTimestamp().DiffMilliseconds(start): the
//            dc-2.0 clock, next to GetWorldTime (TBD-1).
//   e04      one shot, needs DC_Ego. Reads the S105's prefab, component classes, gearbox, assistance mode and engine
//            facts (e04_vehicle.txt, e04_components.csv). Read only: nothing on the vehicle is changed.
//   e20      one shot. Dumps every variable of the engine and game user-settings modules generically
//            (e20_settings.csv) and the display facts (e20_display.txt). Read only: no setting is changed or saved.
//   e19      file-writer cases c01..c11, one per frame, c05 (a write through a READ handle) last. Each case is also
//            logged on the console, so the result survives even if the CSV writer itself cannot be trusted. c10
//            rewrites a 17-byte file with 5 bytes through a new WRITE handle (does WRITE truncate? the dc-2.0
//            heartbeat), c11 appends one line per open-APPEND-write-close cycle (the dc-2.0 commit-log pattern).
//   e19kill  opt-in, never in the default plan. Writes chunk files and two acknowledged commit logs for a manual
//            "taskkill /F" during the kill window: e19_commit.csv through one handle kept open, and
//            e19_commit_append.csv opened with APPEND, written and closed for every line (the dc-2.0 pattern);
//            e19_write.py --kill checks both afterwards. A chunk or commit line that fails is reported to the
//            context, so the runner ends the probe with write_fail.
//
// Every number that could depend on the build is an attribute (echoed as params.* into each run header); no clock
// is a CallLater: e19kill paces itself on world time in OnProbeFrame.
//
// Failure handling: a missing controller, simulation or settings module is recorded as empty values or a
// "missing_module" row and logged; a file that cannot be written fails the probe start (footer status failed).
// OnProbeStop only drops references: these probes change nothing that needs restoring, so "delete" is safe.
// ============================================================================

[ComponentEditorProps(category: "GameScripted/DataCapture", description: "T0-ENG probes E-04 (gearbox, components), E-10 (ticks), E-19 (writer semantics) and E-20 (render settings)")]
class DC_ProbeEngineClass : DC_ProbeComponentClass
{
}

//------------------------------------------------------------------------------------------------
class DC_ProbeEngine : DC_ProbeComponent
{
	static const string ID_E04 = "e04";
	static const string ID_E10 = "e10";
	static const string ID_E19 = "e19";
	static const string ID_E19KILL = "e19kill";
	static const string ID_E20 = "e20";

	//! E-19 case order: c05 runs last, only it could take the process down
	static const int E19_CASE_COUNT = 11;
	//! c11: open-APPEND-write-close cycles after the first line
	static const int E19_APPEND_CYCLES = 3;
	static const string E19_ROW = "0123456789abcdef\n";
	static const string E19_RO_DIR = "$profile:DC_Probes/e19";
	static const string E19_RO_TARGET = "$profile:DC_Probes/e19/ro_target.txt";
	//! e19kill payload: "p" + key.ToString(9) + 54 filler characters = 64 characters
	static const string E19_KILL_FILLER = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx";
	static const float E19_KILL_BANNER_MS = 10000;

	// === PROBES ===
	[Attribute("0.05", UIWidgets.Slider, "E-10: telemetry period [s]; k = max(1, round(period / GetFixedTimeSlice()))", "0.001 1 0.001", category: "Probes")]
	protected float m_fE10TelemetryPeriodS;

	[Attribute("1", UIWidgets.CheckBox, "E-10: write e10_steps.csv, one row per physics step (pre and post call of the same step)", category: "Probes")]
	protected bool m_bE10StepRows;

	[Attribute("1048576", UIWidgets.Slider, "E-19: size of the largest single write of case c03 [bytes]", "1024 16777216 1024", category: "Probes")]
	protected int m_iE19LargeBytes;

	[Attribute("300", UIWidgets.Slider, "e19kill: kill window [world s]; the probe ends normally when nobody killed Workbench by then", "10 3600 10", category: "Probes")]
	protected float m_fE19KillWindowS;

	[Attribute("0.1", UIWidgets.Slider, "e19kill: one chunk file every [world s]", "0.02 10 0.01", category: "Probes")]
	protected float m_fE19KillChunkS;

	[Attribute("6", UIWidgets.Slider, "e19kill: rows per chunk file", "1 1000 1", category: "Probes")]
	protected int m_iE19KillRowsPerChunk;

	[Attribute("VideoUserSettings,DisplayUserSettings", UIWidgets.EditBox, "E-20: engine user-settings modules to dump (GetEngineUserSettings().GetModule), comma separated", category: "Probes")]
	protected string m_sE20EngineModules;

	[Attribute("SCR_VideoSettings", UIWidgets.EditBox, "E-20: game user-settings modules to dump (GetGameUserSettings().GetModule), comma separated", category: "Probes")]
	protected string m_sE20GameModules;

	// === STATE ===
	protected ref DC_ProbeRow m_Row = new DC_ProbeRow();

	// E-10
	protected ref DC_ProbeContext m_E10Context;
	protected ref DC_ProbeCsv m_E10Frames;
	protected ref DC_ProbeCsv m_E10Steps;
	protected ref DC_ProbeCsv m_E10Telemetry;
	protected VehicleWheeledSimulation m_E10EgoSimulation;
	protected int m_iE10K = 1;
	protected int m_iE10LastSimulate;
	protected int m_iE10LastPostSimulate;
	protected int m_iE10LastFixedFrame;
	protected int m_iE10LastPostFixedFrame;
	protected int m_iE10PreStep = -1;			//!< step number of the last pre call not yet written, -1 = none
	protected float m_fE10PreWorldMs;
	protected float m_fE10PreTimeSliceS;
	protected int m_iE10PreRenderFrame;
	protected int m_iE10LastTelemetryStep = -1;
	protected WorldTimestamp m_E10StartTimestamp;	//!< BaseWorld.GetTimestamp() at the start of e10
	protected bool m_bE10HasTimestamp;

	// E-04, E-20
	protected bool m_bE04Done;
	protected bool m_bE20Done;

	// E-19
	protected ref DC_ProbeContext m_E19Context;
	protected ref DC_ProbeCsv m_E19Cases;
	protected int m_iE19NextCase;
	protected string m_sE19RowPath;			//!< the c01 file, reused by c08 and c05

	// e19kill
	protected ref DC_ProbeContext m_KillContext;
	protected ref DC_ProbeCsv m_KillCommit;
	protected string m_sKillAppendPath;		//!< e19_commit_append.csv: one open-APPEND-write-close cycle per line
	protected string m_sKillChunkDir;
	protected int m_iKillChunkId;
	protected int m_iKillKey;
	protected float m_fKillStartMs;
	protected float m_fKillNextChunkMs;
	protected float m_fKillNextBannerMs;
	protected bool m_bKillDone;

	//------------------------------------------------------------------------------------------------
	override void GetProbeIds(notnull array<string> outIds)
	{
		outIds.Insert(ID_E04);
		outIds.Insert(ID_E10);
		outIds.Insert(ID_E19);
		outIds.Insert(ID_E19KILL);
		outIds.Insert(ID_E20);
	}

	//------------------------------------------------------------------------------------------------
	override bool IsConcurrent(string probeId)
	{
		return probeId == ID_E10;
	}

	//------------------------------------------------------------------------------------------------
	override bool NeedsEgo(string probeId)
	{
		return probeId == ID_E04;
	}

	//------------------------------------------------------------------------------------------------
	override float GetProbeTimeoutS(string probeId)
	{
		if (probeId == ID_E10)
			return 7200;

		if (probeId == ID_E19)
			return 60;

		if (probeId == ID_E19KILL)
			return m_fE19KillWindowS + 30;

		return 30;	// e04, e20: one shot
	}

	//------------------------------------------------------------------------------------------------
	//! Every attribute of the component (named as in the layer), plus k and dt_s for e10
	override void GetProbeParams(string probeId, notnull array<string> outKeyValues)
	{
		outKeyValues.Insert("m_fE10TelemetryPeriodS=" + m_fE10TelemetryPeriodS.ToString(-1, 4));
		outKeyValues.Insert("m_bE10StepRows=" + DC_LaunchGuard.BoolText(m_bE10StepRows));
		outKeyValues.Insert("m_iE19LargeBytes=" + m_iE19LargeBytes.ToString());
		outKeyValues.Insert("m_fE19KillWindowS=" + m_fE19KillWindowS.ToString(-1, 1));
		outKeyValues.Insert("m_fE19KillChunkS=" + m_fE19KillChunkS.ToString(-1, 3));
		outKeyValues.Insert("m_iE19KillRowsPerChunk=" + m_iE19KillRowsPerChunk.ToString());
		outKeyValues.Insert("m_sE20EngineModules=" + m_sE20EngineModules);
		outKeyValues.Insert("m_sE20GameModules=" + m_sE20GameModules);

		if (probeId == ID_E10)
		{
			float dt = GetFixedTimeSliceS();
			int k = ComputeTelemetryK();
			outKeyValues.Insert("dt_s=" + dt.ToString(-1, 6));
			outKeyValues.Insert("k=" + k.ToString());
		}
	}

	//------------------------------------------------------------------------------------------------
	override bool OnProbeStart(string probeId, notnull DC_ProbeContext ctx)
	{
		if (probeId == ID_E10)
			return StartE10(ctx);

		if (probeId == ID_E04)
			return RunE04(ctx);

		if (probeId == ID_E20)
			return RunE20(ctx);

		if (probeId == ID_E19)
			return StartE19(ctx);

		if (probeId == ID_E19KILL)
			return StartE19Kill(ctx);

		return false;
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (probeId != ID_E10 || !m_E10Context)
			return;

		// Kept until the post call of the same step writes the e10_steps row
		m_iE10PreStep = physicsStep;
		m_fE10PreTimeSliceS = timeSlice;
		DC_ProbeRunner runner = m_E10Context.GetRunner();
		if (!runner)
			return;

		m_fE10PreWorldMs = runner.GetWorldTimeMs();
		m_iE10PreRenderFrame = runner.GetRenderFrame();
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbePostSimulate(string probeId, float timeSlice, int physicsStep)
	{
		if (probeId != ID_E10 || !m_E10Context)
			return;

		DC_ProbeRunner runner = m_E10Context.GetRunner();
		if (!runner)
			return;

		float worldMs = runner.GetWorldTimeMs();
		if (m_E10Steps)
		{
			// A post without its own pre call keeps the last number and gets empty pre fields: visible in E-10
			m_Row.Clear();
			m_Row.AddInt(physicsStep);
			if (m_iE10PreStep == physicsStep)
			{
				m_Row.AddFloat(m_fE10PreWorldMs, 3);
				m_Row.AddFloat(m_fE10PreTimeSliceS, 6);
				m_Row.AddInt(m_iE10PreRenderFrame);
			}
			else
			{
				m_Row.AddEmpty();
				m_Row.AddEmpty();
				m_Row.AddEmpty();
			}

			m_Row.AddFloat(worldMs, 3);
			m_Row.AddFloat(timeSlice, 6);
			m_Row.AddInt(runner.GetRenderFrame());
			m_Row.AddFloat(GetE10TimestampMs(), 3);
			m_E10Steps.WriteRow(m_Row);
			m_iE10PreStep = -1;
		}

		if (!m_E10Telemetry || physicsStep % m_iE10K != 0 || physicsStep == m_iE10LastTelemetryStep)
			return;

		m_iE10LastTelemetryStep = physicsStep;
		float speedKmh = -1;
		if (m_E10EgoSimulation)
			speedKmh = m_E10EgoSimulation.GetSpeedKmh();

		m_Row.Clear();
		m_Row.AddInt(physicsStep);
		m_Row.AddFloat(worldMs, 3);
		m_Row.AddInt(m_iE10K);
		m_Row.AddFloat(speedKmh, 3);
		m_Row.AddFloat(GetE10TimestampMs(), 3);
		m_E10Telemetry.WriteRow(m_Row);
	}

	//------------------------------------------------------------------------------------------------
	//! BaseWorld.GetTimestamp().DiffMilliseconds(start of e10) [ms] (BaseWorld.c:247, WorldTimestamp.c:30; both in
	//! 1.9): the clock dc-2.0 takes world_time_ms from. -1 without a world or a start timestamp
	protected float GetE10TimestampMs()
	{
		BaseWorld world = GetGame().GetWorld();
		if (!world || !m_bE10HasTimestamp)
			return -1;

		WorldTimestamp now = world.GetTimestamp();
		return now.DiffMilliseconds(m_E10StartTimestamp);
	}

	//------------------------------------------------------------------------------------------------
	override void OnProbeFrame(string probeId, float timeSlice, int renderFrame)
	{
		if (probeId == ID_E10)
			WriteE10Frame(timeSlice, renderFrame);
		else if (probeId == ID_E19)
			RunNextE19Case();
		else if (probeId == ID_E19KILL)
			UpdateE19Kill();
	}

	//------------------------------------------------------------------------------------------------
	override bool IsProbeDone(string probeId)
	{
		if (probeId == ID_E10)
			return false;	// concurrent: the runner stops it when the plan ends

		if (probeId == ID_E04)
			return m_bE04Done;

		if (probeId == ID_E20)
			return m_bE20Done;

		if (probeId == ID_E19)
			return m_iE19NextCase >= E19_CASE_COUNT;

		if (probeId == ID_E19KILL)
			return m_bKillDone;

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Idempotent. Nothing to restore: the files belong to the context, which closes them; only references are dropped.
	override void OnProbeStop(string probeId, string reason)
	{
		if (probeId == ID_E10)
		{
			m_E10Frames = null;
			m_E10Steps = null;
			m_E10Telemetry = null;
			m_E10EgoSimulation = null;
			m_E10Context = null;
		}
		else if (probeId == ID_E19)
		{
			m_E19Cases = null;
			m_E19Context = null;
		}
		else if (probeId == ID_E19KILL)
		{
			m_KillCommit = null;
			m_KillContext = null;
		}
	}

	//------------------------------------------------------------------------------------------------
	//! BaseWorld.GetFixedTimeSlice(), 0 without a world
	protected float GetFixedTimeSliceS()
	{
		ArmaReforgerScripted game = GetGame();
		if (!game)
			return 0;

		BaseWorld world = game.GetWorld();
		if (!world)
			return 0;

		return world.GetFixedTimeSlice();
	}

	//------------------------------------------------------------------------------------------------
	//! k = max(1, round(m_fE10TelemetryPeriodS / GetFixedTimeSlice())) (TBD-1)
	protected int ComputeTelemetryK()
	{
		float dt = GetFixedTimeSliceS();
		if (dt <= 0)
			return 1;

		int k = Math.Round(m_fE10TelemetryPeriodS / dt);
		if (k < 1)
			k = 1;

		return k;
	}

	//------------------------------------------------------------------------------------------------
	// E-10
	//------------------------------------------------------------------------------------------------
	protected bool StartE10(notnull DC_ProbeContext ctx)
	{
		m_E10Context = ctx;
		m_iE10K = ComputeTelemetryK();
		m_iE10PreStep = -1;
		m_iE10LastTelemetryStep = -1;
		m_bE10HasTimestamp = false;
		BaseWorld world = GetGame().GetWorld();
		if (world)
		{
			m_E10StartTimestamp = world.GetTimestamp();
			m_bE10HasTimestamp = true;
		}

		array<string> frameColumns = {"render_frame", "world_time_ms", "tick_ms", "frame_timeslice_s", "fixed_steps", "n_simulate", "n_postsimulate", "n_fixedframe", "n_postfixedframe", "simulate_total", "postsimulate_total", "fixedframe_total", "postfixedframe_total", "fps_avg", "paused", "step_source"};
		m_E10Frames = ctx.OpenCsv("e10_frames", frameColumns);
		if (!m_E10Frames)
			return false;

		if (m_bE10StepRows)
		{
			array<string> stepColumns = {"physics_step", "world_time_ms_pre", "timeslice_pre_s", "render_frame_pre", "world_time_ms_post", "timeslice_post_s", "render_frame_post", "world_ts_ms_post"};
			m_E10Steps = ctx.OpenCsv("e10_steps", stepColumns);
			if (!m_E10Steps)
				return false;
		}

		array<string> telemetryColumns = {"physics_step", "world_time_ms", "k", "speed_kmh", "world_ts_ms"};
		m_E10Telemetry = ctx.OpenCsv("e10_telemetry", telemetryColumns);
		if (!m_E10Telemetry)
			return false;

		IEntity ego = ctx.GetEgo();
		if (ego)
			m_E10EgoSimulation = VehicleWheeledSimulation.Cast(ego.FindComponent(VehicleWheeledSimulation));

		DC_ProbeRunner runner = ctx.GetRunner();
		if (runner)
		{
			m_iE10LastSimulate = runner.GetSimulateCount();
			m_iE10LastPostSimulate = runner.GetPostSimulateCount();
			m_iE10LastFixedFrame = runner.GetFixedFrameCount();
			m_iE10LastPostFixedFrame = runner.GetPostFixedFrameCount();
		}

		float dt = GetFixedTimeSliceS();
		ctx.Log(string.Format("E-10: fixed time slice %1 s, telemetry every k=%2 steps, step rows %3, ego speed %4", dt.ToString(-1, 6), m_iE10K, m_bE10StepRows, m_E10EgoSimulation != null));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! One e10_frames row: the runner's counters and their deltas since the previous frame
	protected void WriteE10Frame(float timeSlice, int renderFrame)
	{
		if (!m_E10Context || !m_E10Frames)
			return;

		DC_ProbeRunner runner = m_E10Context.GetRunner();
		ArmaReforgerScripted game = GetGame();
		if (!runner || !game)
			return;

		BaseWorld world = game.GetWorld();
		if (!world)
			return;

		int simulate = runner.GetSimulateCount();
		int postSimulate = runner.GetPostSimulateCount();
		int fixedFrame = runner.GetFixedFrameCount();
		int postFixedFrame = runner.GetPostFixedFrameCount();

		m_Row.Clear();
		m_Row.AddInt(renderFrame);
		m_Row.AddFloat(runner.GetWorldTimeMs(), 3);
		m_Row.AddInt(runner.GetTickMs());
		m_Row.AddFloat(timeSlice, 6);
		m_Row.AddInt(world.GetFixedSteps());
		m_Row.AddInt(simulate - m_iE10LastSimulate);
		m_Row.AddInt(postSimulate - m_iE10LastPostSimulate);
		m_Row.AddInt(fixedFrame - m_iE10LastFixedFrame);
		m_Row.AddInt(postFixedFrame - m_iE10LastPostFixedFrame);
		m_Row.AddInt(simulate);
		m_Row.AddInt(postSimulate);
		m_Row.AddInt(fixedFrame);
		m_Row.AddInt(postFixedFrame);
		m_Row.AddFloat(System.GetFPS(), 3);
		m_Row.AddBool(runner.IsPaused());
		m_Row.AddString(runner.GetStepSource());
		m_E10Frames.WriteRow(m_Row);

		m_iE10LastSimulate = simulate;
		m_iE10LastPostSimulate = postSimulate;
		m_iE10LastFixedFrame = fixedFrame;
		m_iE10LastPostFixedFrame = postFixedFrame;
	}

	//------------------------------------------------------------------------------------------------
	// E-04
	//------------------------------------------------------------------------------------------------
	//! One shot: e04_vehicle.txt (key=value) and e04_components.csv. Read only.
	protected bool RunE04(notnull DC_ProbeContext ctx)
	{
		m_bE04Done = true;
		IEntity ego = ctx.GetEgo();
		if (!ego)
		{
			ctx.Error("E-04: no ego entity");
			return false;
		}

		CarControllerComponent controller = CarControllerComponent.Cast(ego.FindComponent(CarControllerComponent));
		VehicleWheeledSimulation simulation;
		if (controller)
			simulation = controller.GetSimulation();	// GetWheeledSimulation is obsolete in 1.8

		if (!simulation)
			simulation = VehicleWheeledSimulation.Cast(ego.FindComponent(VehicleWheeledSimulation));

		array<string> lines = {};
		string prefab;
		EntityPrefabData prefabData = ego.GetPrefabData();
		if (prefabData)
			prefab = prefabData.GetPrefabName();

		lines.Insert("ego_name=" + ego.GetName());
		lines.Insert("prefab=" + prefab);
		lines.Insert("entity_class=" + ego.ClassName());
		lines.Insert("has_controller=" + DC_LaunchGuard.BoolText(controller != null));
		lines.Insert("has_simulation=" + DC_LaunchGuard.BoolText(simulation != null));

		// Global player assistance mode: FULL / PARTIAL shift automatically (EVehicleDrivingAssistanceMode.c:13-21)
		EVehicleDrivingAssistanceMode assistance = BaseVehicleControllerComponent.GetDrivingAssistanceMode();
		lines.Insert("driving_assistance_mode=" + IntText(assistance));
		lines.Insert("driving_assistance_mode_name=" + typename.EnumToString(EVehicleDrivingAssistanceMode, assistance));

		if (controller)
		{
			lines.Insert("has_automatic_gearbox=" + DC_LaunchGuard.BoolText(controller.HasAutomaticGearbox()));
			lines.Insert("current_gear=" + IntText(controller.GetCurrentGear()));
			lines.Insert("future_gear=" + IntText(controller.GetFutureGear()));
			lines.Insert("engine_on=" + DC_LaunchGuard.BoolText(controller.IsEngineOn()));
			lines.Insert("persistent_handbrake=" + DC_LaunchGuard.BoolText(controller.GetPersistentHandBrake()));
			lines.Insert("controller_handbrake=" + DC_LaunchGuard.BoolText(controller.GetHandBrake()));
		}
		else
		{
			ctx.Warn("E-04: no CarControllerComponent on " + ego.GetName() + " - gearbox keys left empty");
			lines.Insert("has_automatic_gearbox=");
			lines.Insert("current_gear=");
			lines.Insert("future_gear=");
			lines.Insert("engine_on=");
			lines.Insert("persistent_handbrake=");
			lines.Insert("controller_handbrake=");
		}

		if (simulation)
		{
			lines.Insert("gearbox_gears=" + IntText(simulation.GearboxGearsCount()));
			lines.Insert("gearbox_forward_gears=" + IntText(simulation.GearboxForwardGearsCount()));
			lines.Insert("gear_now=" + IntText(simulation.GetGear()));
			lines.Insert("gearbox_efficiency_state=" + FloatText(simulation.GearboxGetEfficiencyState(), 4));
			lines.Insert("sim_engine_on=" + DC_LaunchGuard.BoolText(simulation.EngineIsOn()));
			lines.Insert("rpm_idle=" + FloatText(simulation.EngineGetRPMIdle(), 1));
			lines.Insert("rpm_max=" + FloatText(simulation.EngineGetRPMMax(), 1));
			lines.Insert("rpm_peak_power=" + FloatText(simulation.EngineGetRPMPeakPower(), 1));
			lines.Insert("handbrake_on=" + DC_LaunchGuard.BoolText(simulation.IsHandbrakeOn()));
			lines.Insert("noise_steer_sensitivity=" + FloatText(simulation.GetNoiseSteerSensitivity(), 4));
			lines.Insert("roughness_sensitivity=" + FloatText(simulation.GetRoughnessSensitivity(), 4));

			int wheelCount = simulation.WheelCount();
			lines.Insert("wheel_count=" + IntText(wheelCount));
			for (int i = 0; i < wheelCount; i++)
			{
				lines.Insert("wheel." + i.ToString() + ".name=" + simulation.WheelGetName(i));
			}
		}
		else
		{
			ctx.Warn("E-04: no VehicleWheeledSimulation on " + ego.GetName() + " - gearbox and engine keys left empty");
			lines.Insert("gearbox_gears=");
			lines.Insert("gearbox_forward_gears=");
			lines.Insert("wheel_count=");
		}

		Physics physics = ego.GetPhysics();
		if (physics)
			lines.Insert("mass_kg=" + FloatText(physics.GetMass(), 2));
		else
			lines.Insert("mass_kg=");

		// Local bounds, for the R1 camera mount (DC_ProbeImages.m_vMountPos); same notation as a layer vector
		vector boundsMin;
		vector boundsMax;
		ego.GetBounds(boundsMin, boundsMax);
		lines.Insert("bounds_min=" + VectorText(boundsMin, 3));
		lines.Insert("bounds_max=" + VectorText(boundsMax, 3));

		array<Managed> components = {};
		ego.FindComponents(GenericComponent, components);
		lines.Insert("component_count=" + IntText(components.Count()));

		if (!ctx.WriteTextFile("e04_vehicle.txt", lines))
			return false;

		array<string> columns = {"idx", "class_name"};
		DC_ProbeCsv csv = ctx.OpenCsv("e04_components", columns);
		if (!csv)
			return false;

		for (int c = 0, count = components.Count(); c < count; c++)
		{
			string className = "null";
			if (components[c])
				className = components[c].ClassName();

			m_Row.Clear();
			m_Row.AddInt(c);
			m_Row.AddString(DC_ProbeCsv.CleanField(className));
			if (!csv.WriteRow(m_Row))
				return false;
		}

		string gearbox = "unknown (no controller)";
		if (controller)
			gearbox = DC_LaunchGuard.BoolText(controller.HasAutomaticGearbox());

		ctx.Log(string.Format("E-04: %1 (%2): %3 components, automatic gearbox %4, assistance %5", ego.GetName(), prefab, components.Count(), gearbox, typename.EnumToString(EVehicleDrivingAssistanceMode, assistance)));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	// E-20
	//------------------------------------------------------------------------------------------------
	//! One shot: e20_settings.csv (every variable of every listed module) and e20_display.txt. Read only.
	protected bool RunE20(notnull DC_ProbeContext ctx)
	{
		m_bE20Done = true;
		ArmaReforgerScripted game = GetGame();

		array<string> columns = {"root", "module", "var_index", "var_name", "var_type", "value"};
		DC_ProbeCsv csv = ctx.OpenCsv("e20_settings", columns);
		if (!csv)
			return false;

		array<string> display = {};
		if (!DumpSettings(ctx, csv, "engine", game.GetEngineUserSettings(), m_sE20EngineModules, display))
			return false;

		if (!DumpSettings(ctx, csv, "game", game.GetGameUserSettings(), m_sE20GameModules, display))
			return false;

		WorkspaceWidget workspace = game.GetWorkspace();
		if (workspace)
		{
			int width = workspace.GetWidth();
			int height = workspace.GetHeight();
			// GetWidth/GetHeight are the render-target size in physical pixels (WorkspaceWidget.c:37-40); the
			// reference-unit size is DPIUnscale of it, as vanilla computes it (SCR_ActionMenuInteractionDisplay.c:507)
			display.Insert("workspace_w=" + IntText(width));
			display.Insert("workspace_h=" + IntText(height));
			display.Insert("workspace_ref_w=" + FloatText(workspace.DPIUnscale(width), 2));
			display.Insert("workspace_ref_h=" + FloatText(workspace.DPIUnscale(height), 2));
			display.Insert("dpi_scale=" + FloatText(workspace.DPIScale(1), 6));
			display.Insert("dpi_user=" + FloatText(workspace.GetUserDPIScale(), 4));
		}
		else
		{
			ctx.Warn("E-20: no workspace widget - workspace keys left empty");
			display.Insert("workspace_w=");
			display.Insert("workspace_h=");
		}

		int renderingWidth;
		int renderingHeight;
		System.GetRenderingResolution(renderingWidth, renderingHeight);
		int nativeWidth;
		int nativeHeight;
		System.GetNativeResolution(nativeWidth, nativeHeight);
		display.Insert("rendering_w=" + IntText(renderingWidth));
		display.Insert("rendering_h=" + IntText(renderingHeight));
		display.Insert("native_w=" + IntText(nativeWidth));
		display.Insert("native_h=" + IntText(nativeHeight));
		display.Insert("fps_avg=" + FloatText(System.GetFPS(), 3));

		string maxFps;
		if (!System.GetCLIParam("maxFPS", maxFps))
			maxFps = string.Empty;

		display.Insert("cli.maxFPS=" + maxFps);
		display.Insert("cli.forceUpdate=" + DC_LaunchGuard.BoolText(System.IsCLIParam("forceUpdate")));

		if (!ctx.WriteTextFile("e20_display.txt", display))
			return false;

		ctx.Log(string.Format("E-20: %1 settings rows; rendering %2x%3, native %4x%5", csv.GetRowCount(), renderingWidth, renderingHeight, nativeWidth, nativeHeight));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Dumps every variable of every module in moduleList (comma separated) of settings. A missing module gets one
	//! row with var_index -1 and var_type missing_module. Adds module.<root>.<name>.class / .vars to display.
	//! \return false only when a CSV write failed
	protected bool DumpSettings(notnull DC_ProbeContext ctx, notnull DC_ProbeCsv csv, string root, UserSettings settings, string moduleList, notnull array<string> display)
	{
		array<string> names = {};
		moduleList.Split(",", names, true);
		foreach (string rawName : names)
		{
			string moduleName = rawName.Trim();
			if (moduleName.IsEmpty())
				continue;

			BaseContainer module;
			if (settings)
				module = settings.GetModule(moduleName);

			string displayKey = "module." + root + "." + moduleName;
			if (!module)
			{
				ctx.Warn("E-20: no " + root + " settings module '" + moduleName + "'");
				display.Insert(displayKey + ".class=");
				display.Insert(displayKey + ".vars=-1");
				m_Row.Clear();
				m_Row.AddString(root);
				m_Row.AddString(DC_ProbeCsv.CleanField(moduleName));
				m_Row.AddInt(-1);
				m_Row.AddEmpty();
				m_Row.AddString("missing_module");
				m_Row.AddEmpty();
				if (!csv.WriteRow(m_Row))
					return false;

				continue;
			}

			int varCount = module.GetNumVars();
			display.Insert(displayKey + ".class=" + module.GetClassName());
			display.Insert(displayKey + ".vars=" + IntText(varCount));
			for (int i = 0; i < varCount; i++)
			{
				string varName = module.GetVarName(i);
				DataVarType varType = module.GetDataVarType(i);
				m_Row.Clear();
				m_Row.AddString(root);
				m_Row.AddString(DC_ProbeCsv.CleanField(moduleName));
				m_Row.AddInt(i);
				m_Row.AddString(DC_ProbeCsv.CleanField(varName));
				m_Row.AddString(typename.EnumToString(DataVarType, varType));
				m_Row.AddString(ReadSettingValue(module, varName, varType));
				if (!csv.WriteRow(m_Row))
					return false;
			}

			ctx.Log(string.Format("E-20: %1 %2 (%3): %4 variables", root, moduleName, module.GetClassName(), varCount));
		}

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! The value as CSV-safe text for SCALAR, INTEGER, FLAGS, BOOLEAN and STRING; empty for every other type
	protected string ReadSettingValue(notnull BaseContainer module, string varName, DataVarType varType)
	{
		if (varType == DataVarType.SCALAR)
		{
			float scalar;
			if (module.Get(varName, scalar))
				return scalar.ToString(-1, 6);
		}
		else if (varType == DataVarType.INTEGER || varType == DataVarType.FLAGS)
		{
			int integer;
			if (module.Get(varName, integer))
				return integer.ToString();
		}
		else if (varType == DataVarType.BOOLEAN)
		{
			bool flag;
			if (module.Get(varName, flag))
				return DC_LaunchGuard.BoolText(flag);
		}
		else if (varType == DataVarType.STRING)
		{
			string text;
			if (module.Get(varName, text))
				return DC_ProbeCsv.CleanField(text);
		}

		return string.Empty;
	}

	//------------------------------------------------------------------------------------------------
	// E-19
	//------------------------------------------------------------------------------------------------
	protected bool StartE19(notnull DC_ProbeContext ctx)
	{
		m_E19Context = ctx;
		m_iE19NextCase = 0;
		m_sE19RowPath = ctx.GetPlayDir() + "/c01_ascii_row.txt";

		array<string> columns = {"case_id", "description", "path", "mode", "open_ok", "bytes_requested", "bytes_returned", "pos_after", "length_after", "expected", "detected", "verdict"};
		m_E19Cases = ctx.OpenCsv("e19_cases", columns);
		if (!m_E19Cases)
			return false;

		ctx.Log("E-19: 11 writer cases, one per frame; c05 (write through a READ handle) runs last");
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Runs the next case: c01 c02 c03 c04 c06 c07 c08 c09 c10 c11, then c05
	protected void RunNextE19Case()
	{
		if (!m_E19Context || m_iE19NextCase >= E19_CASE_COUNT)
			return;

		int index = m_iE19NextCase;
		m_iE19NextCase++;
		string dir = m_E19Context.GetPlayDir();
		if (index == 0)
			RunE19AsciiRow();
		else if (index == 1)
			RunE19LengthParam(dir);
		else if (index == 2)
			RunE19Sizes(dir);
		else if (index == 3)
			RunE19WriteLine(dir);
		else if (index == 4)
			RunE19MissingDir(dir);
		else if (index == 5)
			RunE19DirAsFile(dir);
		else if (index == 6)
			RunE19SecondWriter();
		else if (index == 7)
			RunE19ReadOnlyTarget();
		else if (index == 8)
			RunE19OverwriteShorter(dir);
		else if (index == 9)
			RunE19AppendReopen(dir);
		else
			RunE19ReadHandle();
	}

	//------------------------------------------------------------------------------------------------
	//! c01_ascii_row: new file WRITE, Write("0123456789abcdef\n"): returns 17, length 17
	protected void RunE19AsciiRow()
	{
		int requested = E19_ROW.Length();
		FileHandle file = FileIO.OpenFile(m_sE19RowPath, FileMode.WRITE);
		if (!file)
		{
			WriteE19Case("c01_ascii_row", "new_file_write_17_bytes", m_sE19RowPath, "WRITE", false, requested, "", "", "", "ret:17/len:17", true, "unexpected");
			return;
		}

		int written = file.Write(E19_ROW);
		int pos = file.GetPos();
		int length = file.GetLength();
		file.Close();
		string verdict = "unexpected";
		if (written == requested && length == requested)
			verdict = "ok";

		WriteE19Case("c01_ascii_row", "new_file_write_17_bytes", m_sE19RowPath, "WRITE", true, requested, IntText(written), IntText(pos), IntText(length), "ret:17/len:17", written != requested, verdict);
	}

	//------------------------------------------------------------------------------------------------
	//! c02_length_param: Write(s, 5) on a 16-byte string returns 5
	protected void RunE19LengthParam(string dir)
	{
		string path = dir + "/c02_length_param.txt";
		FileHandle file = FileIO.OpenFile(path, FileMode.WRITE);
		if (!file)
		{
			WriteE19Case("c02_length_param", "write_5_of_16_bytes", path, "WRITE", false, 5, "", "", "", "ret:5/len:5", true, "unexpected");
			return;
		}

		int written = file.Write("0123456789abcdef", 5);
		int pos = file.GetPos();
		int length = file.GetLength();
		file.Close();
		string verdict = "unexpected";
		if (written == 5 && length == 5)
			verdict = "ok";

		WriteE19Case("c02_length_param", "write_5_of_16_bytes", path, "WRITE", true, 5, IntText(written), IntText(pos), IntText(length), "ret:5/len:5", written != 5, verdict);
	}

	//------------------------------------------------------------------------------------------------
	//! c03_sizes: writes of 1, 1024, 65536 and m_iE19LargeBytes bytes into one file; each return equals the size and
	//! the length is the sum. One row per write (c03_sizes_<n>) and one total row (c03_sizes_total).
	protected void RunE19Sizes(string dir)
	{
		string path = dir + "/c03_sizes.bin";
		array<int> sizes = {1, 1024, 65536, m_iE19LargeBytes};
		int total;
		foreach (int size : sizes)
		{
			total += size;
		}

		FileHandle file = FileIO.OpenFile(path, FileMode.WRITE);
		if (!file)
		{
			WriteE19Case("c03_sizes_total", "four_writes_one_file", path, "WRITE", false, total, "", "", "", "len:" + IntText(total), true, "unexpected");
			return;
		}

		int returnedTotal;
		bool allExact = true;
		foreach (int requested : sizes)
		{
			string data = MakeE19Payload(requested);
			int written = file.Write(data);
			returnedTotal += written;
			string verdict = "unexpected";
			if (written == requested)
				verdict = "ok";
			else
				allExact = false;

			WriteE19Case("c03_sizes_" + IntText(requested), "single_write_of_n_bytes", path, "WRITE", true, requested, IntText(written), IntText(file.GetPos()), IntText(file.GetLength()), "ret:" + IntText(requested), written != requested, verdict);
		}

		int length = file.GetLength();
		file.Close();
		string totalVerdict = "unexpected";
		if (allExact && length == total)
			totalVerdict = "ok";

		WriteE19Case("c03_sizes_total", "four_writes_one_file", path, "WRITE", true, total, IntText(returnedTotal), "", IntText(length), "len:" + IntText(total), !allExact, totalVerdict);
	}

	//------------------------------------------------------------------------------------------------
	//! c04_writeline: WriteLine("abc") is void; the position advance is recorded (CR LF expected), never used for data
	protected void RunE19WriteLine(string dir)
	{
		string path = dir + "/c04_writeline.txt";
		FileHandle file = FileIO.OpenFile(path, FileMode.WRITE);
		if (!file)
		{
			WriteE19Case("c04_writeline", "writeline_abc", path, "WRITE", false, 3, "", "", "", "pos:5_crlf", true, "observed");
			return;
		}

		file.WriteLine("abc");
		int pos = file.GetPos();
		int length = file.GetLength();
		file.Close();
		WriteE19Case("c04_writeline", "writeline_abc", path, "WRITE", true, 3, "", IntText(pos), IntText(length), "pos:5_crlf", false, "observed");
	}

	//------------------------------------------------------------------------------------------------
	//! c06_missing_dir: OpenFile in a directory that does not exist: null (failure detected) or the directory is created
	protected void RunE19MissingDir(string dir)
	{
		string path = dir + "/no_such_dir/x.txt";
		FileHandle file = FileIO.OpenFile(path, FileMode.WRITE);
		if (!file)
		{
			WriteE19Case("c06_missing_dir", "open_in_missing_dir", path, "WRITE", false, 1, "", "", "", "open_null_or_dir_created", true, "ok");
			return;
		}

		int written = file.Write("x");
		int pos = file.GetPos();
		int length = file.GetLength();
		file.Close();
		string description = "open_in_missing_dir_dir_created:" + DC_LaunchGuard.BoolText(FileIO.FileExists(dir + "/no_such_dir"));
		WriteE19Case("c06_missing_dir", description, path, "WRITE", true, 1, IntText(written), IntText(pos), IntText(length), "open_null_or_dir_created", written != 1, "observed");
	}

	//------------------------------------------------------------------------------------------------
	//! c07_dir_as_file: OpenFile on an existing directory: null expected (failure detected)
	protected void RunE19DirAsFile(string dir)
	{
		string path = dir + "/is_a_dir";
		FileIO.MakeDirectory(path);
		FileHandle file = FileIO.OpenFile(path, FileMode.WRITE);
		if (!file)
		{
			WriteE19Case("c07_dir_as_file", "open_directory_as_file", path, "WRITE", false, 1, "", "", "", "open_null_or_short", true, "ok");
			return;
		}

		int written = file.Write("x");
		int pos = file.GetPos();
		int length = file.GetLength();
		file.Close();
		string verdict = "unexpected";
		if (written != 1)
			verdict = "ok";

		WriteE19Case("c07_dir_as_file", "open_directory_as_file", path, "WRITE", true, 1, IntText(written), IntText(pos), IntText(length), "open_null_or_short", written != 1, verdict);
	}

	//------------------------------------------------------------------------------------------------
	//! c08_second_writer: the c01 file opened WRITE twice, one byte through each handle (share mode, observed)
	protected void RunE19SecondWriter()
	{
		FileHandle first = FileIO.OpenFile(m_sE19RowPath, FileMode.WRITE);
		FileHandle second = FileIO.OpenFile(m_sE19RowPath, FileMode.WRITE);

		string firstReturned;
		string firstPos;
		if (first)
		{
			int firstWritten = first.Write("A");
			firstReturned = IntText(firstWritten);
			firstPos = IntText(first.GetPos());
		}

		string secondReturned;
		string secondPos;
		string secondLength;
		if (second)
		{
			int secondWritten = second.Write("B");
			secondReturned = IntText(secondWritten);
			secondPos = IntText(second.GetPos());
			secondLength = IntText(second.GetLength());
		}

		if (first)
			first.Close();

		if (second)
			second.Close();

		WriteE19Case("c08_second_writer_h1", "first_of_two_write_handles", m_sE19RowPath, "WRITE", first != null, 1, firstReturned, firstPos, "", "observed", false, "observed");
		WriteE19Case("c08_second_writer_h2", "second_of_two_write_handles", m_sE19RowPath, "WRITE", second != null, 1, secondReturned, secondPos, secondLength, "observed", false, "observed");
	}

	//------------------------------------------------------------------------------------------------
	//! c09_readonly_target: APPEND 2 bytes to $profile:DC_Probes/e19/ro_target.txt (made read-only by the user between
	//! two Plays: attrib +R). Returns 0 or the open fails (detected); "prepared" when the file was just created.
	protected void RunE19ReadOnlyTarget()
	{
		if (!FileIO.FileExists(E19_RO_TARGET))
		{
			FileIO.MakeDirectory(E19_RO_DIR);
			FileHandle created = FileIO.OpenFile(E19_RO_TARGET, FileMode.WRITE);
			string createdReturned;
			if (created)
			{
				int createdWritten = created.Write("ro");
				createdReturned = IntText(createdWritten);
				created.Close();
			}

			WriteE19Case("c09_readonly_target", "target_created_set_it_read_only", E19_RO_TARGET, "WRITE", created != null, 2, createdReturned, "", "", "ret:0_or_open_null", false, "prepared");
			m_E19Context.Log("c09: created " + E19_RO_TARGET + ". Make it read-only (cmd: attrib +R E:\\DriveDiT\\wb_profile\\profile\\DC_Probes\\e19\\ro_target.txt) and run e19 again in a later Play");
			return;
		}

		FileHandle file = FileIO.OpenFile(E19_RO_TARGET, FileMode.APPEND);
		if (!file)
		{
			WriteE19Case("c09_readonly_target", "append_to_read_only_target", E19_RO_TARGET, "APPEND", false, 2, "", "", "", "ret:0_or_open_null", true, "ok");
			return;
		}

		int written = file.Write("ro");
		int pos = file.GetPos();
		int length = file.GetLength();
		file.Close();
		string verdict = "observed";
		if (written != 2)
			verdict = "ok";
		else
			m_E19Context.Warn("c09: " + E19_RO_TARGET + " accepted the write - it is not read-only (attrib +R), so c09 shows nothing");

		WriteE19Case("c09_readonly_target", "append_to_read_only_target", E19_RO_TARGET, "APPEND", true, 2, IntText(written), IntText(pos), IntText(length), "ret:0_or_open_null", written != 2, verdict);
	}

	//------------------------------------------------------------------------------------------------
	//! c10_overwrite_shorter: a 17-byte file opened again with WRITE and given 5 bytes. The length read back through
	//! a READ handle after the close says whether WRITE truncates (5) or keeps the old tail (17): the dc-2.0
	//! heartbeat rewrites one file every 5 s (TBD-3). Observed, not judged; detected = the 5-byte write came back short
	protected void RunE19OverwriteShorter(string dir)
	{
		string path = dir + "/c10_overwrite_shorter.txt";
		FileHandle first = FileIO.OpenFile(path, FileMode.WRITE);
		if (!first)
		{
			WriteE19Case("c10_overwrite_shorter", "rewrite_17_byte_file_with_5_bytes", path, "WRITE", false, 5, "", "", "", "len:5_truncates_or_17_keeps_tail", true, "unexpected");
			return;
		}

		first.Write(E19_ROW);
		first.Close();

		FileHandle second = FileIO.OpenFile(path, FileMode.WRITE);
		if (!second)
		{
			WriteE19Case("c10_overwrite_shorter", "rewrite_17_byte_file_with_5_bytes", path, "WRITE", false, 5, "", "", "", "len:5_truncates_or_17_keeps_tail", true, "unexpected");
			return;
		}

		int written = second.Write("short");
		int pos = second.GetPos();
		second.Close();

		int lengthClosed = -1;
		FileHandle check = FileIO.OpenFile(path, FileMode.READ);
		if (check)
		{
			lengthClosed = check.GetLength();
			check.Close();
		}

		string verdict = "observed";
		if (lengthClosed == 5)
			verdict = "truncates";
		else if (lengthClosed == E19_ROW.Length())
			verdict = "keeps_tail";

		WriteE19Case("c10_overwrite_shorter", "rewrite_17_byte_file_with_5_bytes", path, "WRITE", true, 5, IntText(written), IntText(pos), IntText(lengthClosed), "len:5_truncates_or_17_keeps_tail", written != 5, verdict);
	}

	//------------------------------------------------------------------------------------------------
	//! c11_append_reopen: one 17-byte line written with WRITE, then E19_APPEND_CYCLES cycles of OpenFile(APPEND),
	//! Write(line), Close (the dc-2.0 commit-log pattern). Every return is 17 and the length read back is
	//! (1 + cycles) x 17: verdict ok; anything else is detected
	protected void RunE19AppendReopen(string dir)
	{
		string path = dir + "/c11_append_reopen.txt";
		int lineBytes = E19_ROW.Length();
		int expectedLength = (1 + E19_APPEND_CYCLES) * lineBytes;
		FileHandle create = FileIO.OpenFile(path, FileMode.WRITE);
		if (!create)
		{
			WriteE19Case("c11_append_reopen", "append_reopen_close_per_line", path, "APPEND", false, expectedLength, "", "", "", "len:" + IntText(expectedLength), true, "unexpected");
			return;
		}

		int returnedTotal = create.Write(E19_ROW);
		create.Close();

		bool allExact = returnedTotal == lineBytes;
		bool allOpened = true;
		for (int cycle = 0; cycle < E19_APPEND_CYCLES; cycle++)
		{
			FileHandle append = FileIO.OpenFile(path, FileMode.APPEND);
			if (!append)
			{
				allOpened = false;
				allExact = false;
				continue;
			}

			int written = append.Write(E19_ROW);
			append.Close();
			returnedTotal += written;
			if (written != lineBytes)
				allExact = false;
		}

		int lengthClosed = -1;
		FileHandle check = FileIO.OpenFile(path, FileMode.READ);
		if (check)
		{
			lengthClosed = check.GetLength();
			check.Close();
		}

		string verdict = "unexpected";
		if (allExact && lengthClosed == expectedLength)
			verdict = "ok";

		WriteE19Case("c11_append_reopen", "append_reopen_close_per_line", path, "APPEND", allOpened, expectedLength, IntText(returnedTotal), "", IntText(lengthClosed), "len:" + IntText(expectedLength), !allExact, verdict);
	}

	//------------------------------------------------------------------------------------------------
	//! c05_read_handle: Write("x") through a READ handle of the c01 file returns 0 (detected). Runs last.
	protected void RunE19ReadHandle()
	{
		FileHandle file = FileIO.OpenFile(m_sE19RowPath, FileMode.READ);
		if (!file)
		{
			WriteE19Case("c05_read_handle", "write_through_read_handle", m_sE19RowPath, "READ", false, 1, "", "", "", "ret:0", false, "skipped");
			return;
		}

		int written = file.Write("x");
		int pos = file.GetPos();
		int length = file.GetLength();
		file.Close();
		string verdict = "unexpected";
		if (written != 1)
			verdict = "ok";

		WriteE19Case("c05_read_handle", "write_through_read_handle", m_sE19RowPath, "READ", true, 1, IntText(written), IntText(pos), IntText(length), "ret:0", written != 1, verdict);
	}

	//------------------------------------------------------------------------------------------------
	//! One e19_cases row, also logged on the console (the console line is the fallback record of E-19)
	protected void WriteE19Case(string caseId, string description, string path, string mode, bool openOk, int requested, string returned, string posAfter, string lengthAfter, string expected, bool detected, string verdict)
	{
		if (m_E19Context)
			m_E19Context.Log(string.Format("%1 open=%2 requested=%3 returned=%4 pos=%5 length=%6 detected=%7 verdict=%8", caseId, DC_LaunchGuard.BoolText(openOk), requested, returned, posAfter, lengthAfter, DC_LaunchGuard.BoolText(detected), verdict));

		if (!m_E19Cases)
			return;

		m_Row.Clear();
		m_Row.AddString(caseId);
		m_Row.AddString(DC_ProbeCsv.CleanField(description));
		m_Row.AddString(DC_ProbeCsv.CleanField(path));
		m_Row.AddString(mode);
		m_Row.AddBool(openOk);
		m_Row.AddInt(requested);
		m_Row.AddString(returned);
		m_Row.AddString(posAfter);
		m_Row.AddString(lengthAfter);
		m_Row.AddString(expected);
		m_Row.AddBool(detected);
		m_Row.AddString(verdict);
		m_E19Cases.WriteRow(m_Row);
	}

	//------------------------------------------------------------------------------------------------
	//! size bytes of 'x', built by doubling
	protected string MakeE19Payload(int size)
	{
		string block = "x";
		while (block.Length() * 2 <= size)
		{
			block += block;
		}

		string data;
		while (data.Length() < size)
		{
			int missing = size - data.Length();
			if (missing >= block.Length())
				data += block;
			else
				data += block.Substring(0, missing);
		}

		return data;
	}

	//------------------------------------------------------------------------------------------------
	// e19kill
	//------------------------------------------------------------------------------------------------
	protected bool StartE19Kill(notnull DC_ProbeContext ctx)
	{
		m_KillContext = ctx;
		m_bKillDone = false;
		m_iKillChunkId = 0;
		m_iKillKey = 0;
		m_sKillChunkDir = ctx.MakeSubdir("chunks");
		if (m_sKillChunkDir.IsEmpty())
			return false;

		array<string> columns = {"chunk_id", "first_key", "last_key", "n_rows", "n_bytes"};
		m_KillCommit = ctx.OpenCsv("e19_commit", columns);
		if (!m_KillCommit)
			return false;

		// The second commit log: header now, then one open-APPEND-write-close cycle per commit line
		m_sKillAppendPath = ctx.GetPlayDir() + "/e19_commit_append.csv";
		array<string> appendHeader = {"chunk_id,first_key,last_key,n_rows,n_bytes"};
		string failure;
		if (!DC_ProbeCsv.WriteLines(m_sKillAppendPath, appendHeader, false, failure))
		{
			ctx.Error("e19_commit_append.csv: " + failure);
			return false;
		}

		DC_ProbeRunner runner = ctx.GetRunner();
		if (!runner)
			return false;

		m_fKillStartMs = runner.GetWorldTimeMs();
		m_fKillNextChunkMs = m_fKillStartMs;
		m_fKillNextBannerMs = m_fKillStartMs;
		ctx.Log(string.Format("e19kill: a chunk of %1 rows every %2 s, each closed before its commit lines (e19_commit.csv kept open, e19_commit_append.csv reopened per line); kill window %3 s", m_iE19KillRowsPerChunk, m_fE19KillChunkS.ToString(-1, 3), m_fE19KillWindowS.ToString(-1, 0)));
		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Paced on world time: the kill banner every 10 s, a chunk every m_fE19KillChunkS (at most one per frame)
	protected void UpdateE19Kill()
	{
		if (!m_KillContext || m_bKillDone)
			return;

		DC_ProbeRunner runner = m_KillContext.GetRunner();
		if (!runner)
			return;

		float now = runner.GetWorldTimeMs();
		float leftS = m_fE19KillWindowS - (now - m_fKillStartMs) / 1000;
		if (leftS <= 0)
		{
			m_bKillDone = true;
			m_KillContext.Log(string.Format("kill window closed without a kill: %1 chunks committed", m_iKillChunkId));
			return;
		}

		if (now >= m_fKillNextBannerMs)
		{
			m_fKillNextBannerMs += E19_KILL_BANNER_MS;
			m_KillContext.Log(string.Format("kill window open (%1 s left): taskkill /F /IM ArmaReforgerWorkbenchSteamDiag.exe", leftS.ToString(-1, 0)));
		}

		if (now < m_fKillNextChunkMs)
			return;

		float chunkMs = Math.Max(m_fE19KillChunkS, 0.01) * 1000;
		m_fKillNextChunkMs += chunkMs;
		if (m_fKillNextChunkMs < now)
			m_fKillNextChunkMs = now + chunkMs;

		WriteKillChunk(now);
	}

	//------------------------------------------------------------------------------------------------
	//! chunks/chunk_NNNNN.csv (key, world_time_ms, payload), closed, then its commit line in e19_commit.csv. A chunk
	//! that could not be written completely is never committed.
	protected void WriteKillChunk(float worldMs)
	{
		m_iKillChunkId++;
		string path = m_sKillChunkDir + "/chunk_" + m_iKillChunkId.ToString(5) + ".csv";
		DC_ProbeCsv chunk = new DC_ProbeCsv();
		array<string> columns = {"key", "world_time_ms", "payload"};
		bool ok = chunk.Open(path, columns);
		int firstKey = m_iKillKey + 1;
		for (int r = 0; ok && r < m_iE19KillRowsPerChunk; r++)
		{
			m_iKillKey++;
			m_Row.Clear();
			m_Row.AddInt(m_iKillKey);
			m_Row.AddFloat(worldMs, 3);
			m_Row.AddString("p" + m_iKillKey.ToString(9) + E19_KILL_FILLER);
			ok = chunk.WriteRow(m_Row);
		}

		chunk.Close();
		if (!ok)
		{
			// Not opened through the context: report it, so the runner stops the probe with write_fail
			string chunkFailure = "chunk " + IntText(m_iKillChunkId) + " not committed: " + chunk.GetFailure();
			m_KillContext.Error(chunkFailure);
			m_KillContext.ReportWriteFailure(chunkFailure);
			m_bKillDone = true;
			return;
		}

		int rowCount = chunk.GetRowCount();
		int byteCount = chunk.GetByteCount();
		m_Row.Clear();
		m_Row.AddInt(m_iKillChunkId);
		m_Row.AddInt(firstKey);
		m_Row.AddInt(m_iKillKey);
		m_Row.AddInt(rowCount);
		m_Row.AddInt(byteCount);
		m_KillCommit.WriteRow(m_Row);

		string commitText = string.Format("%1,%2,%3,%4,%5", m_iKillChunkId, firstKey, m_iKillKey, rowCount, byteCount);
		array<string> commitLine = {commitText};
		string failure;
		if (!DC_ProbeCsv.WriteLines(m_sKillAppendPath, commitLine, true, failure))
		{
			m_KillContext.Error("e19_commit_append.csv: " + failure);
			m_KillContext.ReportWriteFailure("e19_commit_append.csv: " + failure);
			m_bKillDone = true;
		}
	}

	//------------------------------------------------------------------------------------------------
	// Text helpers
	//------------------------------------------------------------------------------------------------
	protected static string IntText(int value)
	{
		return value.ToString();
	}

	//------------------------------------------------------------------------------------------------
	protected static string FloatText(float value, int decimals)
	{
		return value.ToString(-1, decimals);
	}

	//------------------------------------------------------------------------------------------------
	//! "x y z", the notation of a vector in a layer file
	protected static string VectorText(vector value, int decimals)
	{
		return FloatText(value[0], decimals) + " " + FloatText(value[1], decimals) + " " + FloatText(value[2], decimals);
	}
}
