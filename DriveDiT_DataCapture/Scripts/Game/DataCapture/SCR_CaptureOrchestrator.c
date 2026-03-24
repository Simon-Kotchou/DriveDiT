/**
 * SCR_CaptureOrchestrator.c
 *
 * Central coordinator for all DriveDiT data capture components.
 * Manages capture sessions, synchronizes timestamps, monitors health,
 * and provides unified control over the capture pipeline.
 *
 * This component is the main entry point for the data capture system
 * and should be attached to the GameMode entity.
 */

//------------------------------------------------------------------------------------------------
//! Capture profile enumeration - determines which data streams are active
enum ECaptureProfile
{
	MINIMAL,        //!< Telemetry only - lightweight for long recordings
	RESEARCH,       //!< Telemetry + Depth + Scene - balanced for development
	PRODUCTION      //!< All streams at maximum quality - for final datasets
}

//------------------------------------------------------------------------------------------------
//! Capture session state
enum ECaptureState
{
	STOPPED,        //!< Not capturing
	STARTING,       //!< Initialization in progress
	RUNNING,        //!< Actively capturing
	PAUSED,         //!< Temporarily paused
	STOPPING,       //!< Shutdown in progress
	ERROR           //!< Error state - capture halted
}

//------------------------------------------------------------------------------------------------
//! Component health status
enum EComponentHealth
{
	HEALTHY,        //!< Component operating normally
	DEGRADED,       //!< Component running with reduced functionality
	FAILED          //!< Component has failed
}

//------------------------------------------------------------------------------------------------
//! Session statistics structure
class SCR_CaptureSessionStats
{
	int m_iFramesCaptured;
	int m_iFramesDropped;
	float m_fAverageFrameTimeMs;
	float m_fPeakFrameTimeMs;
	float m_fTotalDataSizeMB;
	float m_fSessionDurationSec;
	int m_iVehiclesTracked;
	int m_iObjectsEnumerated;

	void Reset()
	{
		m_iFramesCaptured = 0;
		m_iFramesDropped = 0;
		m_fAverageFrameTimeMs = 0;
		m_fPeakFrameTimeMs = 0;
		m_fTotalDataSizeMB = 0;
		m_fSessionDurationSec = 0;
		m_iVehiclesTracked = 0;
		m_iObjectsEnumerated = 0;
	}
}

//------------------------------------------------------------------------------------------------
[ComponentEditorProps(category: "DriveDiT/DataCapture", description: "Central coordinator for all data capture components")]
class SCR_CaptureOrchestratorClass : SCR_BaseGameModeComponentClass
{
}

//------------------------------------------------------------------------------------------------
//! Main orchestrator component for DriveDiT data capture
class SCR_CaptureOrchestrator : SCR_BaseGameModeComponent
{
	//------------------------------------------------------------------------------------------------
	// Configuration
	//------------------------------------------------------------------------------------------------

	[Attribute("1", UIWidgets.ComboBox, "Capture profile", "", ParamEnumArray.FromEnum(ECaptureProfile))]
	protected ECaptureProfile m_eCaptureProfile;

	[Attribute("1", UIWidgets.CheckBox, "Auto-start capture when game begins")]
	protected bool m_bAutoStartCapture;

	[Attribute("100", UIWidgets.EditBox, "Base capture interval in milliseconds")]
	protected int m_iBaseCaptureIntervalMs;

	[Attribute("DriveDiT_Captures", UIWidgets.EditBox, "Output directory name (relative to profile)")]
	protected string m_sOutputDirectory;

	[Attribute("1", UIWidgets.CheckBox, "Enable health monitoring")]
	protected bool m_bEnableHealthMonitoring;

	[Attribute("5.0", UIWidgets.EditBox, "Health check interval in seconds")]
	protected float m_fHealthCheckIntervalSec;

	[Attribute("3", UIWidgets.EditBox, "Maximum consecutive failures before component marked as failed")]
	protected int m_iMaxConsecutiveFailures;

	//------------------------------------------------------------------------------------------------
	// Runtime state
	//------------------------------------------------------------------------------------------------

	protected ECaptureState m_eState = ECaptureState.STOPPED;
	protected string m_sSessionId;
	protected string m_sSessionPath;
	protected float m_fSessionStartTime;
	protected float m_fLastHealthCheckTime;
	protected ref SCR_CaptureSessionStats m_SessionStats;

	//------------------------------------------------------------------------------------------------
	// Component references
	//------------------------------------------------------------------------------------------------

	protected SCR_MLDataCollector m_DataCollector;
	protected SCR_AIDrivingSimulator m_DrivingSimulator;
	protected SCR_DepthRaycaster m_DepthRaycaster;
	protected SCR_SceneEnumerator m_SceneEnumerator;
	protected SCR_BinarySerializer m_BinarySerializer;
	protected SCR_DrivingSimDebugUI m_DebugUI;

	//------------------------------------------------------------------------------------------------
	// Component health tracking
	//------------------------------------------------------------------------------------------------

	protected ref map<string, EComponentHealth> m_ComponentHealth;
	protected ref map<string, int> m_ComponentFailureCount;

	//------------------------------------------------------------------------------------------------
	// Singleton access
	//------------------------------------------------------------------------------------------------

	protected static SCR_CaptureOrchestrator s_Instance;

	static SCR_CaptureOrchestrator GetInstance()
	{
		return s_Instance;
	}

	//------------------------------------------------------------------------------------------------
	// Initialization
	//------------------------------------------------------------------------------------------------

	override void OnPostInit(IEntity owner)
	{
		super.OnPostInit(owner);

		s_Instance = this;

		m_SessionStats = new SCR_CaptureSessionStats();
		m_ComponentHealth = new map<string, EComponentHealth>();
		m_ComponentFailureCount = new map<string, int>();

		// Defer component discovery until game mode starts
		Print("[CaptureOrchestrator] Initialized, waiting for game start", LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	override void OnGameModeStart()
	{
		super.OnGameModeStart();

		Print("[CaptureOrchestrator] Game mode started, discovering components...", LogLevel.NORMAL);

		// Discover capture components
		DiscoverComponents();

		// Apply profile settings
		ApplyProfile(m_eCaptureProfile);

		// Auto-start if configured
		if (m_bAutoStartCapture)
		{
			GetGame().GetCallqueue().CallLater(StartCapture, 1000, false);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Discover and cache references to all capture components
	protected void DiscoverComponents()
	{
		IEntity owner = GetOwner();
		if (!owner)
			return;

		// Find MLDataCollector
		m_DataCollector = SCR_MLDataCollector.Cast(owner.FindComponent(SCR_MLDataCollector));
		if (m_DataCollector)
		{
			m_ComponentHealth.Set("MLDataCollector", EComponentHealth.HEALTHY);
			Print("[CaptureOrchestrator] Found MLDataCollector", LogLevel.NORMAL);
		}

		// Find AIDrivingSimulator
		m_DrivingSimulator = SCR_AIDrivingSimulator.Cast(owner.FindComponent(SCR_AIDrivingSimulator));
		if (m_DrivingSimulator)
		{
			m_ComponentHealth.Set("AIDrivingSimulator", EComponentHealth.HEALTHY);
			Print("[CaptureOrchestrator] Found AIDrivingSimulator", LogLevel.NORMAL);
		}

		// Find DepthRaycaster
		m_DepthRaycaster = SCR_DepthRaycaster.Cast(owner.FindComponent(SCR_DepthRaycaster));
		if (m_DepthRaycaster)
		{
			m_ComponentHealth.Set("DepthRaycaster", EComponentHealth.HEALTHY);
			Print("[CaptureOrchestrator] Found DepthRaycaster", LogLevel.NORMAL);
		}

		// Find SceneEnumerator
		m_SceneEnumerator = SCR_SceneEnumerator.Cast(owner.FindComponent(SCR_SceneEnumerator));
		if (m_SceneEnumerator)
		{
			m_ComponentHealth.Set("SceneEnumerator", EComponentHealth.HEALTHY);
			Print("[CaptureOrchestrator] Found SceneEnumerator", LogLevel.NORMAL);
		}

		// Find BinarySerializer
		m_BinarySerializer = SCR_BinarySerializer.Cast(owner.FindComponent(SCR_BinarySerializer));
		if (m_BinarySerializer)
		{
			m_ComponentHealth.Set("BinarySerializer", EComponentHealth.HEALTHY);
			Print("[CaptureOrchestrator] Found BinarySerializer", LogLevel.NORMAL);
		}

		// Find DebugUI
		m_DebugUI = SCR_DrivingSimDebugUI.Cast(owner.FindComponent(SCR_DrivingSimDebugUI));
		if (m_DebugUI)
		{
			m_ComponentHealth.Set("DebugUI", EComponentHealth.HEALTHY);
			Print("[CaptureOrchestrator] Found DebugUI", LogLevel.NORMAL);
		}

		Print(string.Format("[CaptureOrchestrator] Discovered %1 components", m_ComponentHealth.Count()), LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! Apply capture profile settings to all components
	protected void ApplyProfile(ECaptureProfile profile)
	{
		Print(string.Format("[CaptureOrchestrator] Applying profile: %1", typename.EnumToString(ECaptureProfile, profile)), LogLevel.NORMAL);

		switch (profile)
		{
			case ECaptureProfile.MINIMAL:
				ApplyMinimalProfile();
				break;
			case ECaptureProfile.RESEARCH:
				ApplyResearchProfile();
				break;
			case ECaptureProfile.PRODUCTION:
				ApplyProductionProfile();
				break;
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void ApplyMinimalProfile()
	{
		// Minimal: Telemetry only, 5Hz
		m_iBaseCaptureIntervalMs = 200;

		if (m_DataCollector)
			m_DataCollector.SetCaptureInterval(200);

		// Disable optional components
		if (m_DepthRaycaster)
			m_DepthRaycaster.SetEnabled(false);

		if (m_SceneEnumerator)
			m_SceneEnumerator.SetEnabled(false);
	}

	//------------------------------------------------------------------------------------------------
	protected void ApplyResearchProfile()
	{
		// Research: Telemetry + Depth + Scene, 10Hz
		m_iBaseCaptureIntervalMs = 100;

		if (m_DataCollector)
			m_DataCollector.SetCaptureInterval(100);

		if (m_DepthRaycaster)
		{
			m_DepthRaycaster.SetEnabled(true);
			m_DepthRaycaster.SetResolution(64, 48);
		}

		if (m_SceneEnumerator)
		{
			m_SceneEnumerator.SetEnabled(true);
			m_SceneEnumerator.SetMaxRadius(200);
		}
	}

	//------------------------------------------------------------------------------------------------
	protected void ApplyProductionProfile()
	{
		// Production: All streams at maximum quality, 20Hz
		m_iBaseCaptureIntervalMs = 50;

		if (m_DataCollector)
			m_DataCollector.SetCaptureInterval(50);

		if (m_DepthRaycaster)
		{
			m_DepthRaycaster.SetEnabled(true);
			m_DepthRaycaster.SetResolution(128, 96);
		}

		if (m_SceneEnumerator)
		{
			m_SceneEnumerator.SetEnabled(true);
			m_SceneEnumerator.SetMaxRadius(500);
		}
	}

	//------------------------------------------------------------------------------------------------
	// Session Management
	//------------------------------------------------------------------------------------------------

	//! Start a new capture session
	bool StartCapture()
	{
		if (m_eState == ECaptureState.RUNNING)
		{
			Print("[CaptureOrchestrator] Capture already running", LogLevel.WARNING);
			return false;
		}

		m_eState = ECaptureState.STARTING;

		// Generate session ID
		m_sSessionId = GenerateSessionId();
		m_sSessionPath = string.Format("$profile:%1/%2", m_sOutputDirectory, m_sSessionId);

		Print(string.Format("[CaptureOrchestrator] Starting capture session: %1", m_sSessionId), LogLevel.NORMAL);

		// Initialize components
		bool success = true;

		if (m_DataCollector)
		{
			if (!m_DataCollector.StartSession(m_sSessionPath))
			{
				Print("[CaptureOrchestrator] Failed to start MLDataCollector", LogLevel.ERROR);
				success = false;
			}
		}

		if (m_DepthRaycaster && m_DepthRaycaster.IsEnabled())
		{
			if (!m_DepthRaycaster.StartCapture(m_sSessionPath + "/depth"))
			{
				Print("[CaptureOrchestrator] Failed to start DepthRaycaster", LogLevel.ERROR);
				success = false;
			}
		}

		if (m_SceneEnumerator && m_SceneEnumerator.IsEnabled())
		{
			if (!m_SceneEnumerator.StartCapture(m_sSessionPath + "/scene"))
			{
				Print("[CaptureOrchestrator] Failed to start SceneEnumerator", LogLevel.ERROR);
				success = false;
			}
		}

		if (!success)
		{
			m_eState = ECaptureState.ERROR;
			return false;
		}

		// Record session start
		m_fSessionStartTime = GetGame().GetWorld().GetWorldTime() / 1000.0;
		m_SessionStats.Reset();

		// Write session manifest
		WriteSessionManifest();

		m_eState = ECaptureState.RUNNING;
		Print(string.Format("[CaptureOrchestrator] Capture started: %1", m_sSessionPath), LogLevel.NORMAL);

		return true;
	}

	//------------------------------------------------------------------------------------------------
	//! Stop the current capture session
	void StopCapture()
	{
		if (m_eState != ECaptureState.RUNNING && m_eState != ECaptureState.PAUSED)
		{
			Print("[CaptureOrchestrator] No active capture to stop", LogLevel.WARNING);
			return;
		}

		m_eState = ECaptureState.STOPPING;

		Print("[CaptureOrchestrator] Stopping capture session...", LogLevel.NORMAL);

		// Stop all components
		if (m_DataCollector)
			m_DataCollector.StopSession();

		if (m_DepthRaycaster)
			m_DepthRaycaster.StopCapture();

		if (m_SceneEnumerator)
			m_SceneEnumerator.StopCapture();

		// Calculate final statistics
		m_SessionStats.m_fSessionDurationSec = (GetGame().GetWorld().GetWorldTime() / 1000.0) - m_fSessionStartTime;

		// Write session summary
		WriteSessionSummary();

		m_eState = ECaptureState.STOPPED;
		Print(string.Format("[CaptureOrchestrator] Capture stopped. Duration: %.1f sec, Frames: %1",
			m_SessionStats.m_fSessionDurationSec, m_SessionStats.m_iFramesCaptured), LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! Pause capture (maintains session state)
	void PauseCapture()
	{
		if (m_eState != ECaptureState.RUNNING)
			return;

		m_eState = ECaptureState.PAUSED;

		if (m_DataCollector)
			m_DataCollector.PauseCapture();

		Print("[CaptureOrchestrator] Capture paused", LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! Resume paused capture
	void ResumeCapture()
	{
		if (m_eState != ECaptureState.PAUSED)
			return;

		m_eState = ECaptureState.RUNNING;

		if (m_DataCollector)
			m_DataCollector.ResumeCapture();

		Print("[CaptureOrchestrator] Capture resumed", LogLevel.NORMAL);
	}

	//------------------------------------------------------------------------------------------------
	//! Generate unique session ID
	protected string GenerateSessionId()
	{
		// Format: session_<profile>_<timestamp>
		string profileName;
		switch (m_eCaptureProfile)
		{
			case ECaptureProfile.MINIMAL: profileName = "min"; break;
			case ECaptureProfile.RESEARCH: profileName = "res"; break;
			case ECaptureProfile.PRODUCTION: profileName = "prod"; break;
		}

		int timestamp = System.GetUnixTime();
		return string.Format("session_%1_%2", profileName, timestamp);
	}

	//------------------------------------------------------------------------------------------------
	//! Write session manifest JSON
	protected void WriteSessionManifest()
	{
		string manifestPath = m_sSessionPath + "/manifest.json";

		FileHandle file = FileIO.OpenFile(manifestPath, FileMode.WRITE);
		if (!file)
		{
			Print("[CaptureOrchestrator] Failed to write manifest", LogLevel.ERROR);
			return;
		}

		file.WriteLine("{");
		file.WriteLine(string.Format("  \"session_id\": \"%1\",", m_sSessionId));
		file.WriteLine(string.Format("  \"profile\": \"%1\",", typename.EnumToString(ECaptureProfile, m_eCaptureProfile)));
		file.WriteLine(string.Format("  \"capture_interval_ms\": %1,", m_iBaseCaptureIntervalMs));
		file.WriteLine(string.Format("  \"start_time\": %1,", System.GetUnixTime()));
		file.WriteLine("  \"components\": {");
		file.WriteLine(string.Format("    \"telemetry\": %1,", m_DataCollector != null));
		file.WriteLine(string.Format("    \"depth\": %1,", m_DepthRaycaster != null && m_DepthRaycaster.IsEnabled()));
		file.WriteLine(string.Format("    \"scene\": %1", m_SceneEnumerator != null && m_SceneEnumerator.IsEnabled()));
		file.WriteLine("  },");
		file.WriteLine("  \"version\": \"1.0.0\"");
		file.WriteLine("}");

		file.Close();
	}

	//------------------------------------------------------------------------------------------------
	//! Write session summary JSON
	protected void WriteSessionSummary()
	{
		string summaryPath = m_sSessionPath + "/summary.json";

		FileHandle file = FileIO.OpenFile(summaryPath, FileMode.WRITE);
		if (!file)
		{
			Print("[CaptureOrchestrator] Failed to write summary", LogLevel.ERROR);
			return;
		}

		file.WriteLine("{");
		file.WriteLine(string.Format("  \"session_id\": \"%1\",", m_sSessionId));
		file.WriteLine(string.Format("  \"duration_sec\": %.2f,", m_SessionStats.m_fSessionDurationSec));
		file.WriteLine(string.Format("  \"frames_captured\": %1,", m_SessionStats.m_iFramesCaptured));
		file.WriteLine(string.Format("  \"frames_dropped\": %1,", m_SessionStats.m_iFramesDropped));
		file.WriteLine(string.Format("  \"average_frame_time_ms\": %.2f,", m_SessionStats.m_fAverageFrameTimeMs));
		file.WriteLine(string.Format("  \"peak_frame_time_ms\": %.2f,", m_SessionStats.m_fPeakFrameTimeMs));
		file.WriteLine(string.Format("  \"total_data_size_mb\": %.2f,", m_SessionStats.m_fTotalDataSizeMB));
		file.WriteLine(string.Format("  \"vehicles_tracked\": %1,", m_SessionStats.m_iVehiclesTracked));
		file.WriteLine(string.Format("  \"objects_enumerated\": %1,", m_SessionStats.m_iObjectsEnumerated));
		file.WriteLine(string.Format("  \"end_time\": %1", System.GetUnixTime()));
		file.WriteLine("}");

		file.Close();
	}

	//------------------------------------------------------------------------------------------------
	// Health Monitoring
	//------------------------------------------------------------------------------------------------

	//! Update component health status
	void UpdateComponentHealth(string componentName, EComponentHealth health)
	{
		EComponentHealth previousHealth;
		if (m_ComponentHealth.Find(componentName, previousHealth))
		{
			if (previousHealth != health)
			{
				Print(string.Format("[CaptureOrchestrator] Component %1 health changed: %2 -> %3",
					componentName,
					typename.EnumToString(EComponentHealth, previousHealth),
					typename.EnumToString(EComponentHealth, health)), LogLevel.NORMAL);
			}
		}

		m_ComponentHealth.Set(componentName, health);

		// Track failures
		if (health == EComponentHealth.FAILED)
		{
			int failureCount;
			m_ComponentFailureCount.Find(componentName, failureCount);
			m_ComponentFailureCount.Set(componentName, failureCount + 1);
		}
		else
		{
			m_ComponentFailureCount.Set(componentName, 0);
		}
	}

	//------------------------------------------------------------------------------------------------
	//! Get overall system health
	EComponentHealth GetOverallHealth()
	{
		EComponentHealth worstHealth = EComponentHealth.HEALTHY;

		foreach (string name, EComponentHealth health : m_ComponentHealth)
		{
			if (health > worstHealth)
				worstHealth = health;
		}

		return worstHealth;
	}

	//------------------------------------------------------------------------------------------------
	// Accessors
	//------------------------------------------------------------------------------------------------

	ECaptureState GetState() { return m_eState; }
	ECaptureProfile GetProfile() { return m_eCaptureProfile; }
	string GetSessionId() { return m_sSessionId; }
	string GetSessionPath() { return m_sSessionPath; }
	SCR_CaptureSessionStats GetStats() { return m_SessionStats; }

	SCR_MLDataCollector GetDataCollector() { return m_DataCollector; }
	SCR_AIDrivingSimulator GetDrivingSimulator() { return m_DrivingSimulator; }
	SCR_DepthRaycaster GetDepthRaycaster() { return m_DepthRaycaster; }
	SCR_SceneEnumerator GetSceneEnumerator() { return m_SceneEnumerator; }

	//------------------------------------------------------------------------------------------------
	//! Record a captured frame (called by components)
	void RecordFrame(float frameTimeMs)
	{
		m_SessionStats.m_iFramesCaptured++;

		// Update average
		float totalTime = m_SessionStats.m_fAverageFrameTimeMs * (m_SessionStats.m_iFramesCaptured - 1);
		m_SessionStats.m_fAverageFrameTimeMs = (totalTime + frameTimeMs) / m_SessionStats.m_iFramesCaptured;

		// Update peak
		if (frameTimeMs > m_SessionStats.m_fPeakFrameTimeMs)
			m_SessionStats.m_fPeakFrameTimeMs = frameTimeMs;
	}

	//------------------------------------------------------------------------------------------------
	//! Record a dropped frame
	void RecordDroppedFrame()
	{
		m_SessionStats.m_iFramesDropped++;
	}

	//------------------------------------------------------------------------------------------------
	override void OnGameModeEnd(SCR_GameModeEndData data)
	{
		super.OnGameModeEnd(data);

		if (m_eState == ECaptureState.RUNNING || m_eState == ECaptureState.PAUSED)
		{
			StopCapture();
		}
	}

	//------------------------------------------------------------------------------------------------
	void ~SCR_CaptureOrchestrator()
	{
		if (s_Instance == this)
			s_Instance = null;
	}
}
