// ============================================================================
// SCR_ModularCaptureExample - Example Integration of Modular Capture System
// ============================================================================
//
// Demonstrates how to use the new modular capture infrastructure to replace
// the monolithic SCR_MLDataCollector approach.
//
// MIGRATION GUIDE:
// Old: SCR_MLDataCollector (single component, tight coupling)
// New: SCR_CaptureOrchestrator + modular plugins (loose coupling, extensible)
//
// USAGE:
//   1. Add SCR_ModularCaptureIntegration to your GameMode
//   2. Configure modules via editor attributes or code
//   3. Session auto-starts on game mode start (optional)
//
// ============================================================================

// -----------------------------------------------------------------------------
// Example game mode integration component
// -----------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Modular capture system integration")]
class SCR_ModularCaptureIntegrationClass: SCR_BaseGameModeComponentClass
{
}

class SCR_ModularCaptureIntegration: SCR_BaseGameModeComponent
{
    // === CONFIGURATION ===
    [Attribute("1", UIWidgets.CheckBox, "Enable capture system")]
    protected bool m_bEnableCapture;

    [Attribute("1", UIWidgets.CheckBox, "Auto-start session on game mode start")]
    protected bool m_bAutoStart;

    [Attribute("200", UIWidgets.Slider, "Global capture interval (ms)", "50 1000 50")]
    protected int m_iCaptureIntervalMs;

    // Module toggles
    [Attribute("1", UIWidgets.CheckBox, "Enable telemetry module")]
    protected bool m_bEnableTelemetry;

    [Attribute("1", UIWidgets.CheckBox, "Enable depth module")]
    protected bool m_bEnableDepth;

    [Attribute("1", UIWidgets.CheckBox, "Enable scene module")]
    protected bool m_bEnableScene;

    [Attribute("1", UIWidgets.CheckBox, "Enable road module")]
    protected bool m_bEnableRoad;

    // === INTERNAL STATE ===
    protected ref SCR_CaptureOrchestrator m_Orchestrator;
    protected ref SCR_TelemetryModule m_TelemetryModule;
    protected ref SCR_DepthModule m_DepthModule;
    protected ref SCR_SceneModule m_SceneModule;
    protected ref SCR_RoadModule m_RoadModule;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        if (!m_bEnableCapture)
        {
            Print("[ModularCapture] Capture disabled", LogLevel.NORMAL);
            return;
        }

        // Create orchestrator
        m_Orchestrator = SCR_CaptureOrchestrator.Cast(owner.FindComponent(SCR_CaptureOrchestrator));
        if (!m_Orchestrator)
        {
            Print("[ModularCapture] WARNING: No SCR_CaptureOrchestrator found on entity", LogLevel.WARNING);
            Print("[ModularCapture] Add SCR_CaptureOrchestrator component to game mode", LogLevel.WARNING);
            return;
        }

        // Configure orchestrator
        SCR_CaptureConfig config = m_Orchestrator.GetConfig();
        config.SetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, m_iCaptureIntervalMs);

        // Register modules
        RegisterModules();

        // Subscribe to events
        m_Orchestrator.GetOnSessionStarted().Insert(OnSessionStarted);
        m_Orchestrator.GetOnSessionEnded().Insert(OnSessionEnded);
        m_Orchestrator.GetOnFrameCaptured().Insert(OnFrameCaptured);
        m_Orchestrator.GetOnError().Insert(OnCaptureError);

        Print("[ModularCapture] Initialized", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void RegisterModules()
    {
        if (!m_Orchestrator)
            return;

        // Register telemetry module
        if (m_bEnableTelemetry)
        {
            m_TelemetryModule = new SCR_TelemetryModule();
            SCR_CaptureResult result = m_Orchestrator.RegisterModule(m_TelemetryModule);
            if (!result.IsSuccess())
                Print("[ModularCapture] Failed to register telemetry: " + result.GetMessage(), LogLevel.ERROR);
        }

        // Register depth module
        if (m_bEnableDepth)
        {
            m_DepthModule = new SCR_DepthModule();
            SCR_CaptureResult result = m_Orchestrator.RegisterModule(m_DepthModule);
            if (!result.IsSuccess())
                Print("[ModularCapture] Failed to register depth: " + result.GetMessage(), LogLevel.ERROR);
        }

        // Register scene module
        if (m_bEnableScene)
        {
            m_SceneModule = new SCR_SceneModule();
            SCR_CaptureResult result = m_Orchestrator.RegisterModule(m_SceneModule);
            if (!result.IsSuccess())
                Print("[ModularCapture] Failed to register scene: " + result.GetMessage(), LogLevel.ERROR);
        }

        // Register road module
        if (m_bEnableRoad)
        {
            m_RoadModule = new SCR_RoadModule();
            SCR_CaptureResult result = m_Orchestrator.RegisterModule(m_RoadModule);
            if (!result.IsSuccess())
                Print("[ModularCapture] Failed to register road: " + result.GetMessage(), LogLevel.ERROR);
        }

        Print("[ModularCapture] Registered " + m_Orchestrator.GetRegisteredModuleCount().ToString() + " modules", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    override void OnGameModeStart()
    {
        super.OnGameModeStart();

        if (!m_bEnableCapture || !m_Orchestrator)
            return;

        // Find and register vehicles
        RegisterVehicles();

        // Auto-start session if configured
        if (m_bAutoStart)
        {
            // Delay slightly to allow vehicles to initialize
            GetGame().GetCallqueue().CallLater(StartCapture, 2000, false);
        }
    }

    //------------------------------------------------------------------------
    protected void RegisterVehicles()
    {
        if (!m_Orchestrator)
            return;

        // Find AI vehicles
        for (int i = 1; i <= 10; i++)
        {
            string vehicleName = "AI_Car_" + i.ToString();
            IEntity vehicle = GetGame().GetWorld().FindEntityByName(vehicleName);
            if (vehicle)
            {
                m_Orchestrator.AddTarget(vehicle);
            }
        }

        Print("[ModularCapture] Registered " + m_Orchestrator.GetTargetCount().ToString() + " targets", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    void StartCapture()
    {
        if (!m_Orchestrator)
            return;

        SCR_CaptureResult result = m_Orchestrator.StartSession();
        if (!result.IsSuccess())
        {
            Print("[ModularCapture] Failed to start session: " + result.GetMessage(), LogLevel.ERROR);
        }
    }

    //------------------------------------------------------------------------
    void StopCapture()
    {
        if (!m_Orchestrator)
            return;

        m_Orchestrator.EndSession();
    }

    //------------------------------------------------------------------------
    void PauseCapture()
    {
        if (m_Orchestrator)
            m_Orchestrator.PauseSession();
    }

    //------------------------------------------------------------------------
    void ResumeCapture()
    {
        if (m_Orchestrator)
            m_Orchestrator.ResumeSession();
    }

    //------------------------------------------------------------------------
    // Event handlers
    //------------------------------------------------------------------------
    protected void OnSessionStarted(SCR_CaptureSession session)
    {
        Print("[ModularCapture] Session started: " + session.GetSessionId(), LogLevel.NORMAL);
    }

    protected void OnSessionEnded(SCR_CaptureSession session)
    {
        Print("[ModularCapture] Session ended: " + session.GetTotalFrames().ToString() + " frames", LogLevel.NORMAL);
    }

    protected void OnFrameCaptured(SCR_CaptureContext context)
    {
        // Called every frame - use sparingly
        // Good for triggering waypoint updates, etc.
    }

    protected void OnCaptureError(string moduleId, SCR_CaptureResult result)
    {
        Print("[ModularCapture] Error in " + moduleId + ": " + result.GetMessage(), LogLevel.WARNING);
    }

    //------------------------------------------------------------------------
    // Public API for runtime control
    //------------------------------------------------------------------------
    void AddVehicle(IEntity vehicle)
    {
        if (m_Orchestrator)
            m_Orchestrator.AddTarget(vehicle);
    }

    void RemoveVehicle(IEntity vehicle)
    {
        if (m_Orchestrator)
            m_Orchestrator.RemoveTarget(vehicle);
    }

    void SetModuleEnabled(string moduleId, bool enabled)
    {
        if (m_Orchestrator)
            m_Orchestrator.SetModuleEnabled(moduleId, enabled);
    }

    bool IsCapturing()
    {
        if (m_Orchestrator)
            return m_Orchestrator.IsSessionActive();
        return false;
    }

    int GetFrameCount()
    {
        if (m_Orchestrator)
            return m_Orchestrator.GetTotalCaptureCount();
        return 0;
    }

    //------------------------------------------------------------------------
    // Waypoint integration (for telemetry module)
    //------------------------------------------------------------------------
    void SetWaypointClassifications(array<int> types, array<float> speedLimits, array<string> descriptions)
    {
        if (m_TelemetryModule)
            m_TelemetryModule.SetWaypointClassifications(types, speedLimits, descriptions);
    }

    void SetVehicleWaypoint(int vehicleIndex, int waypointIndex)
    {
        if (m_TelemetryModule)
            m_TelemetryModule.SetTargetWaypoint(vehicleIndex, waypointIndex);
    }

    //------------------------------------------------------------------------
    override void OnGameModeEnd(SCR_GameModeEndData data)
    {
        if (m_Orchestrator && m_Orchestrator.IsSessionActive())
        {
            m_Orchestrator.EndSession();
        }

        super.OnGameModeEnd(data);
    }
}

// =============================================================================
// Example: Creating a custom capture module
// =============================================================================

// Custom module for capturing specific data
class SCR_CustomCaptureModule : SCR_ICaptureModule
{
    void SCR_CustomCaptureModule()
    {
        // Define module metadata
        m_Metadata = new SCR_ModuleMetadata(
            "custom",                           // Unique module ID
            "Custom Data",                      // Display name
            "Captures custom application data", // Description
            "1.0.0",                           // Version
            SCR_ModuleCapability.CAP_REAL_TIME | SCR_ModuleCapability.CAP_MULTI_TARGET,
            SCR_CaptureFormat.FORMAT_CSV,
            500,    // Capture every 500ms (2 Hz)
            50      // Priority (lower = higher priority)
        );
    }

    override SCR_CaptureResult Initialize(SCR_CaptureConfig config)
    {
        // Perform any initialization here
        // Access config with: config.GetModuleConfig("custom")
        return super.Initialize(config);
    }

    override string GetCSVHeader()
    {
        return "frame_id,timestamp_ms,target_index,custom_value1,custom_value2";
    }

    override SCR_CaptureResult Capture(SCR_CaptureContext context, SCR_CaptureBuffer buffer)
    {
        // Implement your custom capture logic here
        int frameId = context.GetFrameId();
        float timestampMs = context.GetTimestampMs();

        array<IEntity> targets = context.GetTargets();
        for (int t = 0; t < targets.Count(); t++)
        {
            // Create a custom record
            // buffer.Write(record, timestampMs);
        }

        RecordCapture(timestampMs);
        return SCR_CaptureResult.Success();
    }

    override SCR_CaptureResult Finalize()
    {
        // Cleanup
        return super.Finalize();
    }
}

// =============================================================================
// Migration example: Converting from SCR_MLDataCollector
// =============================================================================
//
// OLD APPROACH (monolithic):
// --------------------------
// SCR_MLDataCollector collector = ...;
// collector.InitializeSession();
// collector.RegisterVehicle(vehicle);
// collector.CaptureFrame();  // Called in update loop
// collector.FinalizeSession();
//
// NEW APPROACH (modular):
// -----------------------
// SCR_CaptureOrchestrator orchestrator = ...;
//
// // Register modules
// orchestrator.RegisterModule(new SCR_TelemetryModule());
// orchestrator.RegisterModule(new SCR_DepthModule());
// orchestrator.RegisterModule(new SCR_SceneModule());
//
// // Add targets
// orchestrator.AddTarget(vehicle);
//
// // Start session (capture runs automatically on timer)
// orchestrator.StartSession();
//
// // ... later ...
// orchestrator.EndSession();
//
// BENEFITS:
// - Add/remove modules at runtime
// - Configure each module independently
// - Ring buffer prevents I/O bottlenecks
// - Multiple output formats (CSV, binary)
// - Clear error handling per module
// - Event-driven architecture
//
