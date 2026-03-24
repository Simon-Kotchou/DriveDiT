// ============================================================================
// SCR_ICaptureModule - Interface for Modular Data Capture
// ============================================================================
//
// Defines the contract for all capture modules in the modular data capture
// architecture. Each module captures a specific type of data (telemetry,
// depth, scene, road, camera, etc.).
//
// IMPLEMENTATION NOTES:
// - Modules should be stateless where possible
// - Use CaptureContext for shared state and configuration
// - Return SCR_CaptureResult for consistent error handling
// - All capture operations should be non-blocking
//
// ============================================================================

// -----------------------------------------------------------------------------
// Error codes for capture operations
// -----------------------------------------------------------------------------
enum SCR_CaptureError
{
    CAPTURE_OK = 0,
    CAPTURE_ERROR_NOT_INITIALIZED,
    CAPTURE_ERROR_INVALID_TARGET,
    CAPTURE_ERROR_BUFFER_FULL,
    CAPTURE_ERROR_IO_FAILURE,
    CAPTURE_ERROR_SERIALIZATION,
    CAPTURE_ERROR_CONFIG_INVALID,
    CAPTURE_ERROR_DEPENDENCY_MISSING,
    CAPTURE_ERROR_RATE_LIMITED,
    CAPTURE_ERROR_DISABLED,
    CAPTURE_ERROR_UNKNOWN
}

// -----------------------------------------------------------------------------
// Module state enumeration
// -----------------------------------------------------------------------------
enum SCR_ModuleState
{
    STATE_UNINITIALIZED = 0,
    STATE_INITIALIZED,
    STATE_CAPTURING,
    STATE_PAUSED,
    STATE_ERROR,
    STATE_FINALIZED
}

// -----------------------------------------------------------------------------
// Data format flags for serialization
// -----------------------------------------------------------------------------
enum SCR_CaptureFormat
{
    FORMAT_CSV = 1,
    FORMAT_BINARY = 2,
    FORMAT_JSON = 4,
    FORMAT_MSGPACK = 8
}

// -----------------------------------------------------------------------------
// Module capability flags
// -----------------------------------------------------------------------------
enum SCR_ModuleCapability
{
    CAP_NONE = 0,
    CAP_ASYNC_CAPTURE = 1,          // Supports async capture
    CAP_BATCH_WRITE = 2,            // Supports batched writes
    CAP_INCREMENTAL = 4,            // Supports incremental capture
    CAP_COMPRESSION = 8,            // Supports data compression
    CAP_MULTI_TARGET = 16,          // Can capture from multiple targets
    CAP_REAL_TIME = 32              // Suitable for real-time capture
}

// -----------------------------------------------------------------------------
// SCR_CaptureResult - Result type for capture operations
// -----------------------------------------------------------------------------
class SCR_CaptureResult
{
    protected SCR_CaptureError m_eError;
    protected string m_sMessage;
    protected int m_iBytesWritten;
    protected float m_fCaptureTimeMs;

    //------------------------------------------------------------------------
    void SCR_CaptureResult()
    {
        m_eError = SCR_CaptureError.CAPTURE_OK;
        m_sMessage = "";
        m_iBytesWritten = 0;
        m_fCaptureTimeMs = 0;
    }

    //------------------------------------------------------------------------
    static SCR_CaptureResult Success(int bytesWritten = 0, float captureTimeMs = 0)
    {
        SCR_CaptureResult result = new SCR_CaptureResult();
        result.m_eError = SCR_CaptureError.CAPTURE_OK;
        result.m_iBytesWritten = bytesWritten;
        result.m_fCaptureTimeMs = captureTimeMs;
        return result;
    }

    //------------------------------------------------------------------------
    static SCR_CaptureResult Failure(SCR_CaptureError error, string message = "")
    {
        SCR_CaptureResult result = new SCR_CaptureResult();
        result.m_eError = error;
        result.m_sMessage = message;
        return result;
    }

    //------------------------------------------------------------------------
    bool IsSuccess() { return m_eError == SCR_CaptureError.CAPTURE_OK; }
    SCR_CaptureError GetError() { return m_eError; }
    string GetMessage() { return m_sMessage; }
    int GetBytesWritten() { return m_iBytesWritten; }
    float GetCaptureTimeMs() { return m_fCaptureTimeMs; }
}

// -----------------------------------------------------------------------------
// SCR_CaptureContext - Shared context passed to all modules during capture
// -----------------------------------------------------------------------------
class SCR_CaptureContext
{
    // Timing
    protected int m_iFrameId;
    protected float m_fTimestampMs;
    protected float m_fDeltaTimeMs;
    protected float m_fSessionStartTimeMs;

    // Session
    protected string m_sSessionId;
    protected string m_sSessionPath;

    // References (set by orchestrator)
    protected ref array<IEntity> m_aTargets;
    protected ref map<string, ref Managed> m_mSharedData;

    // Flags
    protected bool m_bIsCapturing;
    protected int m_iActiveFormats;

    //------------------------------------------------------------------------
    void SCR_CaptureContext()
    {
        m_iFrameId = 0;
        m_fTimestampMs = 0;
        m_fDeltaTimeMs = 0;
        m_fSessionStartTimeMs = 0;
        m_sSessionId = "";
        m_sSessionPath = "";
        m_aTargets = new array<IEntity>();
        m_mSharedData = new map<string, ref Managed>();
        m_bIsCapturing = false;
        m_iActiveFormats = SCR_CaptureFormat.FORMAT_CSV;
    }

    //------------------------------------------------------------------------
    // Timing accessors
    int GetFrameId() { return m_iFrameId; }
    void SetFrameId(int frameId) { m_iFrameId = frameId; }

    float GetTimestampMs() { return m_fTimestampMs; }
    void SetTimestampMs(float timestampMs) { m_fTimestampMs = timestampMs; }

    float GetDeltaTimeMs() { return m_fDeltaTimeMs; }
    void SetDeltaTimeMs(float deltaTimeMs) { m_fDeltaTimeMs = deltaTimeMs; }

    float GetSessionStartTimeMs() { return m_fSessionStartTimeMs; }
    void SetSessionStartTimeMs(float startTimeMs) { m_fSessionStartTimeMs = startTimeMs; }

    float GetElapsedTimeMs() { return m_fTimestampMs - m_fSessionStartTimeMs; }

    //------------------------------------------------------------------------
    // Session accessors
    string GetSessionId() { return m_sSessionId; }
    void SetSessionId(string sessionId) { m_sSessionId = sessionId; }

    string GetSessionPath() { return m_sSessionPath; }
    void SetSessionPath(string sessionPath) { m_sSessionPath = sessionPath; }

    //------------------------------------------------------------------------
    // Target management
    array<IEntity> GetTargets() { return m_aTargets; }

    void AddTarget(IEntity target)
    {
        if (target && m_aTargets.Find(target) < 0)
            m_aTargets.Insert(target);
    }

    void RemoveTarget(IEntity target)
    {
        int idx = m_aTargets.Find(target);
        if (idx >= 0)
            m_aTargets.Remove(idx);
    }

    void ClearTargets() { m_aTargets.Clear(); }
    int GetTargetCount() { return m_aTargets.Count(); }

    IEntity GetTarget(int index)
    {
        if (index >= 0 && index < m_aTargets.Count())
            return m_aTargets[index];
        return null;
    }

    //------------------------------------------------------------------------
    // Shared data (for inter-module communication)
    void SetSharedData(string key, Managed data)
    {
        m_mSharedData.Set(key, data);
    }

    Managed GetSharedData(string key)
    {
        if (m_mSharedData.Contains(key))
            return m_mSharedData.Get(key);
        return null;
    }

    bool HasSharedData(string key)
    {
        return m_mSharedData.Contains(key);
    }

    //------------------------------------------------------------------------
    // State accessors
    bool IsCapturing() { return m_bIsCapturing; }
    void SetCapturing(bool capturing) { m_bIsCapturing = capturing; }

    int GetActiveFormats() { return m_iActiveFormats; }
    void SetActiveFormats(int formats) { m_iActiveFormats = formats; }
    bool HasFormat(SCR_CaptureFormat format) { return (m_iActiveFormats & format) != 0; }
}

// -----------------------------------------------------------------------------
// SCR_ModuleMetadata - Descriptive information about a capture module
// -----------------------------------------------------------------------------
class SCR_ModuleMetadata
{
    protected string m_sModuleId;
    protected string m_sDisplayName;
    protected string m_sDescription;
    protected string m_sVersion;
    protected int m_iCapabilities;
    protected int m_iSupportedFormats;
    protected float m_fDefaultIntervalMs;
    protected int m_iPriority;  // Lower = higher priority (captures first)

    //------------------------------------------------------------------------
    void SCR_ModuleMetadata(
        string moduleId,
        string displayName,
        string description = "",
        string version = "1.0.0",
        int capabilities = SCR_ModuleCapability.CAP_NONE,
        int supportedFormats = SCR_CaptureFormat.FORMAT_CSV,
        float defaultIntervalMs = 200,
        int priority = 100)
    {
        m_sModuleId = moduleId;
        m_sDisplayName = displayName;
        m_sDescription = description;
        m_sVersion = version;
        m_iCapabilities = capabilities;
        m_iSupportedFormats = supportedFormats;
        m_fDefaultIntervalMs = defaultIntervalMs;
        m_iPriority = priority;
    }

    //------------------------------------------------------------------------
    string GetModuleId() { return m_sModuleId; }
    string GetDisplayName() { return m_sDisplayName; }
    string GetDescription() { return m_sDescription; }
    string GetVersion() { return m_sVersion; }
    int GetCapabilities() { return m_iCapabilities; }
    int GetSupportedFormats() { return m_iSupportedFormats; }
    float GetDefaultIntervalMs() { return m_fDefaultIntervalMs; }
    int GetPriority() { return m_iPriority; }

    bool HasCapability(SCR_ModuleCapability cap)
    {
        return (m_iCapabilities & cap) != 0;
    }

    bool SupportsFormat(SCR_CaptureFormat format)
    {
        return (m_iSupportedFormats & format) != 0;
    }
}

// -----------------------------------------------------------------------------
// SCR_CaptureDataRecord - Base class for captured data records
// -----------------------------------------------------------------------------
class SCR_CaptureDataRecord
{
    protected int m_iFrameId;
    protected float m_fTimestampMs;
    protected string m_sModuleId;
    protected int m_iTargetIndex;

    //------------------------------------------------------------------------
    void SCR_CaptureDataRecord(int frameId, float timestampMs, string moduleId, int targetIndex = 0)
    {
        m_iFrameId = frameId;
        m_fTimestampMs = timestampMs;
        m_sModuleId = moduleId;
        m_iTargetIndex = targetIndex;
    }

    //------------------------------------------------------------------------
    int GetFrameId() { return m_iFrameId; }
    float GetTimestampMs() { return m_fTimestampMs; }
    string GetModuleId() { return m_sModuleId; }
    int GetTargetIndex() { return m_iTargetIndex; }

    //------------------------------------------------------------------------
    // Serialization interface - override in subclasses
    string ToCSV() { return ""; }
    void ToBinary(FileHandle file) { }

    static string GetCSVHeader() { return "frame_id,timestamp_ms,module_id,target_index"; }
}

// -----------------------------------------------------------------------------
// SCR_ICaptureModule - Abstract interface for capture modules
// -----------------------------------------------------------------------------
class SCR_ICaptureModule
{
    protected SCR_ModuleState m_eState;
    protected ref SCR_ModuleMetadata m_Metadata;
    protected ref SCR_CaptureConfig m_Config;
    protected float m_fLastCaptureTimeMs;
    protected int m_iTotalCaptureCount;
    protected int m_iErrorCount;

    //------------------------------------------------------------------------
    void SCR_ICaptureModule()
    {
        m_eState = SCR_ModuleState.STATE_UNINITIALIZED;
        m_fLastCaptureTimeMs = 0;
        m_iTotalCaptureCount = 0;
        m_iErrorCount = 0;
    }

    //------------------------------------------------------------------------
    // REQUIRED: Return module metadata
    SCR_ModuleMetadata GetMetadata()
    {
        return m_Metadata;
    }

    //------------------------------------------------------------------------
    // REQUIRED: Initialize the module with configuration
    // Returns: SCR_CaptureResult indicating success or failure
    SCR_CaptureResult Initialize(SCR_CaptureConfig config)
    {
        m_Config = config;
        m_eState = SCR_ModuleState.STATE_INITIALIZED;
        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    // REQUIRED: Capture data for the current frame
    // Returns: SCR_CaptureResult with captured data info
    SCR_CaptureResult Capture(SCR_CaptureContext context, SCR_CaptureBuffer buffer)
    {
        // Base implementation - override in subclasses
        return SCR_CaptureResult.Failure(
            SCR_CaptureError.CAPTURE_ERROR_NOT_INITIALIZED,
            "Capture not implemented"
        );
    }

    //------------------------------------------------------------------------
    // REQUIRED: Finalize and cleanup
    SCR_CaptureResult Finalize()
    {
        m_eState = SCR_ModuleState.STATE_FINALIZED;
        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    // OPTIONAL: Get CSV header for this module's data
    string GetCSVHeader()
    {
        return "";
    }

    //------------------------------------------------------------------------
    // OPTIONAL: Get binary schema version for this module's data
    int GetBinarySchemaVersion()
    {
        return 1;
    }

    //------------------------------------------------------------------------
    // OPTIONAL: Validate a target entity
    bool ValidateTarget(IEntity target)
    {
        return target != null;
    }

    //------------------------------------------------------------------------
    // OPTIONAL: Called when targets are added/removed
    void OnTargetAdded(IEntity target, int targetIndex) { }
    void OnTargetRemoved(IEntity target, int targetIndex) { }

    //------------------------------------------------------------------------
    // OPTIONAL: Pause/resume capture
    void Pause()
    {
        if (m_eState == SCR_ModuleState.STATE_CAPTURING)
            m_eState = SCR_ModuleState.STATE_PAUSED;
    }

    void Resume()
    {
        if (m_eState == SCR_ModuleState.STATE_PAUSED)
            m_eState = SCR_ModuleState.STATE_CAPTURING;
    }

    //------------------------------------------------------------------------
    // State accessors
    SCR_ModuleState GetState() { return m_eState; }
    void SetState(SCR_ModuleState state) { m_eState = state; }

    bool IsInitialized()
    {
        return m_eState >= SCR_ModuleState.STATE_INITIALIZED &&
               m_eState < SCR_ModuleState.STATE_ERROR;
    }

    bool IsCapturing()
    {
        return m_eState == SCR_ModuleState.STATE_CAPTURING;
    }

    //------------------------------------------------------------------------
    // Statistics
    int GetTotalCaptureCount() { return m_iTotalCaptureCount; }
    int GetErrorCount() { return m_iErrorCount; }
    float GetLastCaptureTimeMs() { return m_fLastCaptureTimeMs; }

    //------------------------------------------------------------------------
    // Rate limiting helper
    bool ShouldCapture(float currentTimeMs)
    {
        if (!m_Metadata || !m_Config)
            return false;

        float interval = m_Config.GetModuleInterval(m_Metadata.GetModuleId());
        if (interval <= 0)
            interval = m_Metadata.GetDefaultIntervalMs();

        return (currentTimeMs - m_fLastCaptureTimeMs) >= interval;
    }

    protected void RecordCapture(float timeMs)
    {
        m_fLastCaptureTimeMs = timeMs;
        m_iTotalCaptureCount++;
    }

    protected void RecordError()
    {
        m_iErrorCount++;
    }
}

// -----------------------------------------------------------------------------
// Forward declaration for circular reference
// SCR_CaptureConfig and SCR_CaptureBuffer are defined in their own files
// -----------------------------------------------------------------------------
class SCR_CaptureConfig;
class SCR_CaptureBuffer;
