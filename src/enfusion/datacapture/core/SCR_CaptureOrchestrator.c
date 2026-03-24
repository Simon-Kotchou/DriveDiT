// ============================================================================
// SCR_CaptureOrchestrator - Central Coordinator for Modular Data Capture
// ============================================================================
//
// The orchestrator is the main entry point for the capture system. It:
// - Manages module lifecycle (registration, initialization, finalization)
// - Coordinates capture timing across modules
// - Manages targets (vehicles, entities) across all modules
// - Handles session management (create, pause, resume, finalize)
// - Provides unified error handling and logging
//
// USAGE:
//   1. Create orchestrator component on game mode entity
//   2. Register capture modules
//   3. Add targets (vehicles)
//   4. Start session
//   5. Capture runs automatically based on configuration
//
// ============================================================================

// -----------------------------------------------------------------------------
// Orchestrator state
// -----------------------------------------------------------------------------
enum SCR_OrchestratorState
{
    ORCH_UNINITIALIZED,
    ORCH_INITIALIZED,
    ORCH_SESSION_ACTIVE,
    ORCH_SESSION_PAUSED,
    ORCH_ERROR,
    ORCH_FINALIZED
}

// -----------------------------------------------------------------------------
// Session information
// -----------------------------------------------------------------------------
class SCR_CaptureSession
{
    protected string m_sSessionId;
    protected string m_sSessionPath;
    protected float m_fStartTimeMs;
    protected float m_fEndTimeMs;
    protected int m_iTotalFrames;
    protected bool m_bActive;

    //------------------------------------------------------------------------
    void SCR_CaptureSession()
    {
        m_sSessionId = "";
        m_sSessionPath = "";
        m_fStartTimeMs = 0;
        m_fEndTimeMs = 0;
        m_iTotalFrames = 0;
        m_bActive = false;
    }

    //------------------------------------------------------------------------
    // Generate unique session ID
    static string GenerateSessionId()
    {
        float worldTime = GetGame().GetWorld().GetWorldTime();
        int timeInt = worldTime;
        return "session_" + timeInt.ToString();
    }

    //------------------------------------------------------------------------
    void Start(string basePath)
    {
        m_sSessionId = GenerateSessionId();
        m_sSessionPath = basePath + "/" + m_sSessionId;
        m_fStartTimeMs = GetGame().GetWorld().GetWorldTime();
        m_iTotalFrames = 0;
        m_bActive = true;

        // Create session directory
        FileIO.MakeDirectory(m_sSessionPath);
    }

    void End()
    {
        m_fEndTimeMs = GetGame().GetWorld().GetWorldTime();
        m_bActive = false;
    }

    //------------------------------------------------------------------------
    string GetSessionId() { return m_sSessionId; }
    string GetSessionPath() { return m_sSessionPath; }
    float GetStartTimeMs() { return m_fStartTimeMs; }
    float GetEndTimeMs() { return m_fEndTimeMs; }
    float GetDurationMs() { return m_fEndTimeMs - m_fStartTimeMs; }
    int GetTotalFrames() { return m_iTotalFrames; }
    bool IsActive() { return m_bActive; }

    void IncrementFrames() { m_iTotalFrames++; }
}

// -----------------------------------------------------------------------------
// Module registration entry
// -----------------------------------------------------------------------------
class SCR_ModuleRegistration
{
    protected ref SCR_ICaptureModule m_Module;
    protected bool m_bEnabled;
    protected int m_iPriority;
    protected float m_fLastCaptureMs;

    //------------------------------------------------------------------------
    void SCR_ModuleRegistration(SCR_ICaptureModule module)
    {
        m_Module = module;
        m_bEnabled = true;
        m_iPriority = 100;
        m_fLastCaptureMs = 0;

        // Get priority from metadata if available
        SCR_ModuleMetadata meta = module.GetMetadata();
        if (meta)
            m_iPriority = meta.GetPriority();
    }

    //------------------------------------------------------------------------
    SCR_ICaptureModule GetModule() { return m_Module; }
    bool IsEnabled() { return m_bEnabled; }
    void SetEnabled(bool enabled) { m_bEnabled = enabled; }
    int GetPriority() { return m_iPriority; }
    float GetLastCaptureMs() { return m_fLastCaptureMs; }
    void SetLastCaptureMs(float timeMs) { m_fLastCaptureMs = timeMs; }
}

// -----------------------------------------------------------------------------
// SCR_CaptureOrchestrator - Main Component
// -----------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Central coordinator for modular data capture system")]
class SCR_CaptureOrchestratorClass: ScriptComponentClass
{
}

class SCR_CaptureOrchestrator: ScriptComponent
{
    // === EDITOR CONFIGURATION ===
    [Attribute("1", UIWidgets.CheckBox, "Enable data capture")]
    protected bool m_bEnableCapture;

    [Attribute("200", UIWidgets.Slider, "Global capture interval (ms)", "50 1000 50")]
    protected int m_iCaptureIntervalMs;

    [Attribute("1", UIWidgets.CheckBox, "Auto-start session on game mode start")]
    protected bool m_bAutoStartSession;

    [Attribute("1", UIWidgets.CheckBox, "Verbose logging")]
    protected bool m_bVerboseLogging;

    [Attribute("500", UIWidgets.Slider, "Progress log interval (frames)", "100 2000 100")]
    protected int m_iProgressLogInterval;

    // === INTERNAL STATE ===
    protected SCR_OrchestratorState m_eState;
    protected ref SCR_CaptureConfig m_Config;
    protected ref SCR_CaptureContext m_Context;
    protected ref SCR_CaptureBuffer m_Buffer;
    protected ref SCR_CaptureSerializer m_Serializer;
    protected ref SCR_CaptureSession m_Session;

    // Module registry
    protected ref array<ref SCR_ModuleRegistration> m_aModules;
    protected ref map<string, int> m_mModuleIndex;

    // Events
    protected ref ScriptInvoker m_OnSessionStarted;
    protected ref ScriptInvoker m_OnSessionEnded;
    protected ref ScriptInvoker m_OnFrameCaptured;
    protected ref ScriptInvoker m_OnError;

    // Statistics
    protected int m_iTotalCaptureCount;
    protected int m_iTotalErrorCount;
    protected float m_fLastCaptureTimeMs;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        // Initialize state
        m_eState = SCR_OrchestratorState.ORCH_UNINITIALIZED;
        m_iTotalCaptureCount = 0;
        m_iTotalErrorCount = 0;
        m_fLastCaptureTimeMs = 0;

        // Initialize collections
        m_aModules = new array<ref SCR_ModuleRegistration>();
        m_mModuleIndex = new map<string, int>();

        // Initialize events
        m_OnSessionStarted = new ScriptInvoker();
        m_OnSessionEnded = new ScriptInvoker();
        m_OnFrameCaptured = new ScriptInvoker();
        m_OnError = new ScriptInvoker();

        // Initialize configuration
        m_Config = new SCR_CaptureConfig();
        m_Config.SetFloat(SCR_ConfigKeys.CAPTURE_INTERVAL_MS, m_iCaptureIntervalMs);
        m_Config.SetBool(SCR_ConfigKeys.LOG_VERBOSE, m_bVerboseLogging);
        m_Config.SetInt(SCR_ConfigKeys.LOG_PROGRESS_INTERVAL, m_iProgressLogInterval);

        // Initialize context
        m_Context = new SCR_CaptureContext();

        // Initialize buffer
        m_Buffer = new SCR_CaptureBuffer(
            m_Config.GetBufferCapacity(),
            SCR_BufferOverflowPolicy.OVERFLOW_DROP_OLDEST
        );

        // Initialize serializer
        m_Serializer = new SCR_CaptureSerializer();
        m_Buffer.SetSerializer(m_Serializer);

        m_eState = SCR_OrchestratorState.ORCH_INITIALIZED;
        Print("[CaptureOrchestrator] Initialized", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Register a capture module
    SCR_CaptureResult RegisterModule(SCR_ICaptureModule module)
    {
        if (!module)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "Module is null");

        SCR_ModuleMetadata meta = module.GetMetadata();
        if (!meta)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "Module has no metadata");

        string moduleId = meta.GetModuleId();

        // Check for duplicate
        if (m_mModuleIndex.Contains(moduleId))
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "Module already registered: " + moduleId);

        // Initialize module
        SCR_CaptureResult initResult = module.Initialize(m_Config);
        if (!initResult.IsSuccess())
            return initResult;

        // Register
        SCR_ModuleRegistration reg = new SCR_ModuleRegistration(module);
        int index = m_aModules.Count();
        m_aModules.Insert(reg);
        m_mModuleIndex.Set(moduleId, index);

        // Sort by priority
        SortModulesByPriority();

        Print("[CaptureOrchestrator] Registered module: " + moduleId + " (priority " + meta.GetPriority().ToString() + ")", LogLevel.NORMAL);
        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    // Unregister a module
    SCR_CaptureResult UnregisterModule(string moduleId)
    {
        if (!m_mModuleIndex.Contains(moduleId))
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "Module not found: " + moduleId);

        int index = m_mModuleIndex.Get(moduleId);
        SCR_ModuleRegistration reg = m_aModules[index];

        // Finalize module
        if (reg && reg.GetModule())
            reg.GetModule().Finalize();

        // Remove from collections
        m_aModules.Remove(index);
        m_mModuleIndex.Remove(moduleId);

        // Rebuild index map
        RebuildModuleIndex();

        Print("[CaptureOrchestrator] Unregistered module: " + moduleId, LogLevel.NORMAL);
        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    // Get a registered module
    SCR_ICaptureModule GetModule(string moduleId)
    {
        if (!m_mModuleIndex.Contains(moduleId))
            return null;

        int index = m_mModuleIndex.Get(moduleId);
        if (index < 0 || index >= m_aModules.Count())
            return null;

        return m_aModules[index].GetModule();
    }

    //------------------------------------------------------------------------
    // Enable/disable a module
    void SetModuleEnabled(string moduleId, bool enabled)
    {
        if (!m_mModuleIndex.Contains(moduleId))
            return;

        int index = m_mModuleIndex.Get(moduleId);
        m_aModules[index].SetEnabled(enabled);

        m_Config.SetModuleEnabled(moduleId, enabled);
    }

    //------------------------------------------------------------------------
    // Add a target entity for capture
    void AddTarget(IEntity target)
    {
        if (!target)
            return;

        m_Context.AddTarget(target);

        // Notify all modules
        int targetIndex = m_Context.GetTargets().Find(target);
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ICaptureModule module = m_aModules[i].GetModule();
            if (module && module.ValidateTarget(target))
            {
                module.OnTargetAdded(target, targetIndex);
            }
        }

        Print("[CaptureOrchestrator] Added target: " + target.GetName() + " (index " + targetIndex.ToString() + ")", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Remove a target entity
    void RemoveTarget(IEntity target)
    {
        if (!target)
            return;

        int targetIndex = m_Context.GetTargets().Find(target);
        if (targetIndex < 0)
            return;

        // Notify all modules
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ICaptureModule module = m_aModules[i].GetModule();
            if (module)
                module.OnTargetRemoved(target, targetIndex);
        }

        m_Context.RemoveTarget(target);
        Print("[CaptureOrchestrator] Removed target: " + target.GetName(), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Start a capture session
    SCR_CaptureResult StartSession()
    {
        if (!m_bEnableCapture)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_DISABLED, "Capture disabled");

        if (m_eState == SCR_OrchestratorState.ORCH_SESSION_ACTIVE)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "Session already active");

        if (m_aModules.IsEmpty())
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "No modules registered");

        // Create session
        m_Session = new SCR_CaptureSession();
        m_Session.Start(m_Config.GetSessionBasePath());

        // Initialize serializer
        SCR_CaptureResult serResult = m_Serializer.Initialize(m_Config, m_Session.GetSessionPath());
        if (!serResult.IsSuccess())
            return serResult;

        // Update context
        m_Context.SetSessionId(m_Session.GetSessionId());
        m_Context.SetSessionPath(m_Session.GetSessionPath());
        m_Context.SetSessionStartTimeMs(m_Session.GetStartTimeMs());
        m_Context.SetCapturing(true);
        m_Context.SetActiveFormats(m_Config.GetOutputFormats());

        // Set module states to capturing
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ICaptureModule module = m_aModules[i].GetModule();
            if (module)
                module.SetState(SCR_ModuleState.STATE_CAPTURING);
        }

        m_eState = SCR_OrchestratorState.ORCH_SESSION_ACTIVE;

        // Write session metadata
        WriteSessionMetadata();

        // Start capture timer
        float intervalMs = m_Config.GetCaptureIntervalMs();
        GetGame().GetCallqueue().CallLater(CaptureFrame, intervalMs, true);

        // Fire event
        m_OnSessionStarted.Invoke(m_Session);

        Print("[CaptureOrchestrator] Session started: " + m_Session.GetSessionId(), LogLevel.NORMAL);
        Print("[CaptureOrchestrator] Capture rate: " + m_Config.GetCaptureRateHz().ToString(5, 1) + " Hz", LogLevel.NORMAL);

        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    // Pause the current session
    void PauseSession()
    {
        if (m_eState != SCR_OrchestratorState.ORCH_SESSION_ACTIVE)
            return;

        m_eState = SCR_OrchestratorState.ORCH_SESSION_PAUSED;
        m_Context.SetCapturing(false);

        // Pause all modules
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ICaptureModule module = m_aModules[i].GetModule();
            if (module)
                module.Pause();
        }

        Print("[CaptureOrchestrator] Session paused", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Resume paused session
    void ResumeSession()
    {
        if (m_eState != SCR_OrchestratorState.ORCH_SESSION_PAUSED)
            return;

        m_eState = SCR_OrchestratorState.ORCH_SESSION_ACTIVE;
        m_Context.SetCapturing(true);

        // Resume all modules
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ICaptureModule module = m_aModules[i].GetModule();
            if (module)
                module.Resume();
        }

        Print("[CaptureOrchestrator] Session resumed", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // End the current session
    SCR_CaptureResult EndSession()
    {
        if (m_eState != SCR_OrchestratorState.ORCH_SESSION_ACTIVE &&
            m_eState != SCR_OrchestratorState.ORCH_SESSION_PAUSED)
        {
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_NOT_INITIALIZED, "No active session");
        }

        // Stop capture timer
        GetGame().GetCallqueue().Remove(CaptureFrame);

        // Flush buffer
        float currentTimeMs = GetGame().GetWorld().GetWorldTime();
        m_Buffer.Finalize(currentTimeMs);

        // Finalize modules
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ICaptureModule module = m_aModules[i].GetModule();
            if (module)
                module.Finalize();
        }

        // Write session summary
        m_Serializer.WriteSessionSummary();
        m_Serializer.Finalize();

        // End session
        m_Session.End();
        m_Context.SetCapturing(false);

        m_eState = SCR_OrchestratorState.ORCH_FINALIZED;

        // Fire event
        m_OnSessionEnded.Invoke(m_Session);

        Print("[CaptureOrchestrator] Session ended: " + m_Session.GetTotalFrames().ToString() + " frames captured", LogLevel.NORMAL);

        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    // Main capture frame - called on timer
    protected void CaptureFrame()
    {
        if (m_eState != SCR_OrchestratorState.ORCH_SESSION_ACTIVE)
            return;

        float currentTimeMs = GetGame().GetWorld().GetWorldTime();

        // Update context
        float deltaTimeMs = currentTimeMs - m_fLastCaptureTimeMs;
        m_Context.SetFrameId(m_iTotalCaptureCount);
        m_Context.SetTimestampMs(currentTimeMs);
        m_Context.SetDeltaTimeMs(deltaTimeMs);

        // Capture from each enabled module
        int modulesCaptuered = 0;
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ModuleRegistration reg = m_aModules[i];
            if (!reg.IsEnabled())
                continue;

            SCR_ICaptureModule module = reg.GetModule();
            if (!module || !module.IsCapturing())
                continue;

            // Check module-specific rate limiting
            if (!module.ShouldCapture(currentTimeMs))
                continue;

            // Perform capture
            SCR_CaptureResult result = module.Capture(m_Context, m_Buffer);

            if (result.IsSuccess())
            {
                reg.SetLastCaptureMs(currentTimeMs);
                modulesCaptuered++;
            }
            else
            {
                HandleModuleError(module, result);
            }
        }

        // Update statistics
        if (modulesCaptuered > 0)
        {
            m_iTotalCaptureCount++;
            m_Session.IncrementFrames();
            m_fLastCaptureTimeMs = currentTimeMs;

            // Fire event
            m_OnFrameCaptured.Invoke(m_Context);
        }

        // Progress logging
        if (m_iTotalCaptureCount % m_iProgressLogInterval == 0)
        {
            LogProgress();
        }
    }

    //------------------------------------------------------------------------
    // Handle module error
    protected void HandleModuleError(SCR_ICaptureModule module, SCR_CaptureResult result)
    {
        m_iTotalErrorCount++;

        SCR_ModuleMetadata meta = module.GetMetadata();
        string moduleId = meta ? meta.GetModuleId() : "unknown";

        Print("[CaptureOrchestrator] Module error: " + moduleId + " - " + result.GetMessage(), LogLevel.WARNING);

        // Fire error event
        m_OnError.Invoke(moduleId, result);
    }

    //------------------------------------------------------------------------
    // Log progress
    protected void LogProgress()
    {
        if (!m_Session)
            return;

        float elapsedMin = (GetGame().GetWorld().GetWorldTime() - m_Session.GetStartTimeMs()) / 60000.0;

        string msg = "[CaptureOrchestrator] ";
        msg += m_iTotalCaptureCount.ToString() + " frames";
        msg += " (" + elapsedMin.ToString(6, 1) + " min)";
        msg += " | Buffer: " + m_Buffer.GetOccupancyPercent().ToString(4, 1) + "%";
        msg += " | Targets: " + m_Context.GetTargetCount().ToString();
        msg += " | Errors: " + m_iTotalErrorCount.ToString();

        Print(msg, LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Write session metadata
    protected void WriteSessionMetadata()
    {
        if (!m_Serializer)
            return;

        m_Serializer.WriteSessionMetadata("session_id", m_Session.GetSessionId());
        m_Serializer.WriteSessionMetadata("start_time_ms", m_Session.GetStartTimeMs().ToString(12, 1));
        m_Serializer.WriteSessionMetadata("capture_interval_ms", m_Config.GetCaptureIntervalMs().ToString());
        m_Serializer.WriteSessionMetadata("capture_rate_hz", m_Config.GetCaptureRateHz().ToString(5, 1));
        m_Serializer.WriteSessionMetadata("target_count", m_Context.GetTargetCount().ToString());
        m_Serializer.WriteSessionMetadata("module_count", m_aModules.Count().ToString());

        // List registered modules
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ICaptureModule module = m_aModules[i].GetModule();
            if (module)
            {
                SCR_ModuleMetadata meta = module.GetMetadata();
                if (meta)
                {
                    m_Serializer.WriteSessionMetadata("module_" + i.ToString(), meta.GetModuleId() + " v" + meta.GetVersion());
                }
            }
        }
    }

    //------------------------------------------------------------------------
    // Sort modules by priority (lower = higher priority)
    protected void SortModulesByPriority()
    {
        // Simple bubble sort (modules list is typically small)
        for (int i = 0; i < m_aModules.Count() - 1; i++)
        {
            for (int j = 0; j < m_aModules.Count() - i - 1; j++)
            {
                if (m_aModules[j].GetPriority() > m_aModules[j + 1].GetPriority())
                {
                    SCR_ModuleRegistration temp = m_aModules[j];
                    m_aModules[j] = m_aModules[j + 1];
                    m_aModules[j + 1] = temp;
                }
            }
        }

        RebuildModuleIndex();
    }

    //------------------------------------------------------------------------
    // Rebuild module index map after sort or removal
    protected void RebuildModuleIndex()
    {
        m_mModuleIndex.Clear();
        for (int i = 0; i < m_aModules.Count(); i++)
        {
            SCR_ICaptureModule module = m_aModules[i].GetModule();
            if (module)
            {
                SCR_ModuleMetadata meta = module.GetMetadata();
                if (meta)
                    m_mModuleIndex.Set(meta.GetModuleId(), i);
            }
        }
    }

    //------------------------------------------------------------------------
    // Event accessors
    ScriptInvoker GetOnSessionStarted() { return m_OnSessionStarted; }
    ScriptInvoker GetOnSessionEnded() { return m_OnSessionEnded; }
    ScriptInvoker GetOnFrameCaptured() { return m_OnFrameCaptured; }
    ScriptInvoker GetOnError() { return m_OnError; }

    //------------------------------------------------------------------------
    // State accessors
    SCR_OrchestratorState GetState() { return m_eState; }
    bool IsSessionActive() { return m_eState == SCR_OrchestratorState.ORCH_SESSION_ACTIVE; }
    SCR_CaptureSession GetSession() { return m_Session; }
    SCR_CaptureConfig GetConfig() { return m_Config; }
    SCR_CaptureContext GetContext() { return m_Context; }
    SCR_CaptureBuffer GetBuffer() { return m_Buffer; }
    SCR_CaptureSerializer GetSerializer() { return m_Serializer; }

    //------------------------------------------------------------------------
    // Statistics
    int GetTotalCaptureCount() { return m_iTotalCaptureCount; }
    int GetTotalErrorCount() { return m_iTotalErrorCount; }
    int GetRegisteredModuleCount() { return m_aModules.Count(); }
    int GetTargetCount() { return m_Context.GetTargetCount(); }

    //------------------------------------------------------------------------
    // Cleanup on destroy
    override void OnDelete(IEntity owner)
    {
        if (m_eState == SCR_OrchestratorState.ORCH_SESSION_ACTIVE ||
            m_eState == SCR_OrchestratorState.ORCH_SESSION_PAUSED)
        {
            EndSession();
        }

        super.OnDelete(owner);
    }

    //------------------------------------------------------------------------
    // Debug output
    string GetDebugString()
    {
        string str = "[Orchestrator] state=" + m_eState.ToString();
        str += " modules=" + m_aModules.Count().ToString();
        str += " targets=" + m_Context.GetTargetCount().ToString();
        str += " frames=" + m_iTotalCaptureCount.ToString();
        str += " errors=" + m_iTotalErrorCount.ToString();

        if (m_Session)
            str += " session=" + m_Session.GetSessionId();

        return str;
    }
}
