// ============================================================================
// SCR_TelemetryModule - Vehicle Telemetry Capture Module
// ============================================================================
//
// Captures vehicle telemetry data including:
// - Position, orientation (forward, up, right vectors)
// - Speed, steering, throttle, brake, clutch
// - Gear, engine RPM, engine state, handbrake
// - Computed: acceleration, total distance
//
// OUTPUT FORMAT (CSV):
//   frame_id, timestamp_ms, target_index, pos_x, pos_y, pos_z, ...
//
// Refactored from SCR_MLDataCollector for modularity.
//
// ============================================================================

// -----------------------------------------------------------------------------
// Telemetry data record
// -----------------------------------------------------------------------------
class SCR_TelemetryRecord : SCR_CaptureDataRecord
{
    // Position
    float m_fPosX;
    float m_fPosY;
    float m_fPosZ;

    // Orientation vectors
    float m_fFwdX, m_fFwdY, m_fFwdZ;
    float m_fUpX, m_fUpY, m_fUpZ;
    float m_fRightX, m_fRightY, m_fRightZ;

    // Vehicle state
    float m_fSpeedKmh;
    float m_fSteering;
    float m_fThrottle;
    float m_fBrake;
    float m_fClutch;
    int m_iGear;
    float m_fEngineRPM;
    bool m_bEngineOn;
    bool m_bHandbrake;

    // Computed
    float m_fAcceleration;
    float m_fDistanceTotal;

    // Context
    int m_iWaypointType;
    float m_fSpeedLimit;
    int m_iWaypointIdx;

    //------------------------------------------------------------------------
    void SCR_TelemetryRecord(int frameId, float timestampMs, int targetIndex)
    {
        // Call parent constructor
        SCR_CaptureDataRecord(frameId, timestampMs, "telemetry", targetIndex);

        // Initialize defaults
        m_fPosX = 0; m_fPosY = 0; m_fPosZ = 0;
        m_fFwdX = 0; m_fFwdY = 0; m_fFwdZ = 1;
        m_fUpX = 0; m_fUpY = 1; m_fUpZ = 0;
        m_fRightX = 1; m_fRightY = 0; m_fRightZ = 0;
        m_fSpeedKmh = 0;
        m_fSteering = 0;
        m_fThrottle = 0;
        m_fBrake = 0;
        m_fClutch = 0;
        m_iGear = 0;
        m_fEngineRPM = 0;
        m_bEngineOn = false;
        m_bHandbrake = false;
        m_fAcceleration = 0;
        m_fDistanceTotal = 0;
        m_iWaypointType = -1;
        m_fSpeedLimit = 0;
        m_iWaypointIdx = -1;
    }

    //------------------------------------------------------------------------
    override string ToCSV()
    {
        string row = "";

        // Frame info
        row += m_iFrameId.ToString() + ",";
        row += m_fTimestampMs.ToString(12, 1) + ",";
        row += m_iTargetIndex.ToString() + ",";

        // Position
        row += m_fPosX.ToString(10, 4) + ",";
        row += m_fPosY.ToString(10, 4) + ",";
        row += m_fPosZ.ToString(10, 4) + ",";

        // Forward vector
        row += m_fFwdX.ToString(8, 6) + ",";
        row += m_fFwdY.ToString(8, 6) + ",";
        row += m_fFwdZ.ToString(8, 6) + ",";

        // Up vector
        row += m_fUpX.ToString(8, 6) + ",";
        row += m_fUpY.ToString(8, 6) + ",";
        row += m_fUpZ.ToString(8, 6) + ",";

        // Right vector
        row += m_fRightX.ToString(8, 6) + ",";
        row += m_fRightY.ToString(8, 6) + ",";
        row += m_fRightZ.ToString(8, 6) + ",";

        // Vehicle state
        row += m_fSpeedKmh.ToString(8, 3) + ",";
        row += m_fSteering.ToString(8, 5) + ",";
        row += m_fThrottle.ToString(8, 5) + ",";
        row += m_fBrake.ToString(8, 5) + ",";
        row += m_fClutch.ToString(8, 5) + ",";
        row += m_iGear.ToString() + ",";
        row += m_fEngineRPM.ToString(8, 1) + ",";

        int engineOnInt = 0;
        if (m_bEngineOn)
            engineOnInt = 1;
        row += engineOnInt.ToString() + ",";

        int handbrakeInt = 0;
        if (m_bHandbrake)
            handbrakeInt = 1;
        row += handbrakeInt.ToString() + ",";

        // Computed
        row += m_fAcceleration.ToString(8, 3) + ",";
        row += m_fDistanceTotal.ToString(10, 2) + ",";

        // Context
        row += m_iWaypointType.ToString() + ",";
        row += m_fSpeedLimit.ToString(6, 1) + ",";
        row += m_iWaypointIdx.ToString();

        return row;
    }

    //------------------------------------------------------------------------
    override void ToBinary(FileHandle file)
    {
        if (!file)
            return;

        // Write fixed-size binary record
        file.Write(m_iFrameId, 4);
        file.Write(m_fTimestampMs, 4);
        file.Write(m_iTargetIndex, 4);

        // Position (12 bytes)
        file.Write(m_fPosX, 4);
        file.Write(m_fPosY, 4);
        file.Write(m_fPosZ, 4);

        // Forward (12 bytes)
        file.Write(m_fFwdX, 4);
        file.Write(m_fFwdY, 4);
        file.Write(m_fFwdZ, 4);

        // Up (12 bytes)
        file.Write(m_fUpX, 4);
        file.Write(m_fUpY, 4);
        file.Write(m_fUpZ, 4);

        // Right (12 bytes)
        file.Write(m_fRightX, 4);
        file.Write(m_fRightY, 4);
        file.Write(m_fRightZ, 4);

        // Vehicle state (24 bytes)
        file.Write(m_fSpeedKmh, 4);
        file.Write(m_fSteering, 4);
        file.Write(m_fThrottle, 4);
        file.Write(m_fBrake, 4);
        file.Write(m_fClutch, 4);
        file.Write(m_iGear, 4);
        file.Write(m_fEngineRPM, 4);

        // Flags (4 bytes)
        int flags = 0;
        if (m_bEngineOn) flags |= 1;
        if (m_bHandbrake) flags |= 2;
        file.Write(flags, 4);

        // Computed (8 bytes)
        file.Write(m_fAcceleration, 4);
        file.Write(m_fDistanceTotal, 4);

        // Context (12 bytes)
        file.Write(m_iWaypointType, 4);
        file.Write(m_fSpeedLimit, 4);
        file.Write(m_iWaypointIdx, 4);
    }

    //------------------------------------------------------------------------
    static override string GetCSVHeader()
    {
        string header = "frame_id,timestamp_ms,target_index,";
        header += "pos_x,pos_y,pos_z,";
        header += "fwd_x,fwd_y,fwd_z,";
        header += "up_x,up_y,up_z,";
        header += "right_x,right_y,right_z,";
        header += "speed_kmh,steering,throttle,brake,clutch,";
        header += "gear,engine_rpm,engine_on,handbrake,";
        header += "acceleration_kmh_s,distance_total_m,";
        header += "waypoint_type,speed_limit_kmh,waypoint_idx";
        return header;
    }
}

// -----------------------------------------------------------------------------
// Per-target tracking state
// -----------------------------------------------------------------------------
class SCR_TelemetryTargetState
{
    IEntity m_Target;
    VehicleWheeledSimulation m_Simulation;
    vector m_vLastPosition;
    float m_fLastSpeed;
    float m_fTotalDistance;
    float m_fAcceleration;
    int m_iCurrentWaypointIdx;
    bool m_bValid;

    //------------------------------------------------------------------------
    void SCR_TelemetryTargetState(IEntity target)
    {
        m_Target = target;
        m_Simulation = null;
        m_vLastPosition = vector.Zero;
        m_fLastSpeed = 0;
        m_fTotalDistance = 0;
        m_fAcceleration = 0;
        m_iCurrentWaypointIdx = -1;
        m_bValid = false;

        if (target)
        {
            m_Simulation = VehicleWheeledSimulation.Cast(target.FindComponent(VehicleWheeledSimulation));
            m_bValid = m_Simulation != null;

            if (m_bValid)
            {
                vector transform[4];
                target.GetWorldTransform(transform);
                m_vLastPosition = transform[3];
            }
        }
    }

    //------------------------------------------------------------------------
    bool IsValid() { return m_bValid && m_Target && m_Simulation; }
}

// -----------------------------------------------------------------------------
// SCR_TelemetryModule - Main telemetry capture module
// -----------------------------------------------------------------------------
class SCR_TelemetryModule : SCR_ICaptureModule
{
    // Target states
    protected ref array<ref SCR_TelemetryTargetState> m_aTargetStates;

    // Waypoint context (shared across targets)
    protected ref array<int> m_aWaypointTypes;
    protected ref array<float> m_aSpeedLimits;
    protected ref array<string> m_aWaypointDescriptions;

    // Configuration
    protected float m_fCaptureIntervalMs;
    protected bool m_bVerboseLogging;

    // CSV writer reference
    protected ref SCR_CSVWriter m_CSVWriter;
    protected bool m_bHeaderWritten;

    //------------------------------------------------------------------------
    void SCR_TelemetryModule()
    {
        // Initialize metadata
        m_Metadata = new SCR_ModuleMetadata(
            "telemetry",                                    // moduleId
            "Vehicle Telemetry",                            // displayName
            "Captures vehicle position, orientation, and control inputs",  // description
            "2.0.0",                                        // version
            SCR_ModuleCapability.CAP_REAL_TIME | SCR_ModuleCapability.CAP_MULTI_TARGET,  // capabilities
            SCR_CaptureFormat.FORMAT_CSV | SCR_CaptureFormat.FORMAT_BINARY,  // supported formats
            200,                                            // default interval (5 Hz)
            10                                              // priority (high - captures first)
        );

        // Initialize collections
        m_aTargetStates = new array<ref SCR_TelemetryTargetState>();
        m_aWaypointTypes = new array<int>();
        m_aSpeedLimits = new array<float>();
        m_aWaypointDescriptions = new array<string>();

        m_bHeaderWritten = false;
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Initialize(SCR_CaptureConfig config)
    {
        SCR_CaptureResult result = super.Initialize(config);
        if (!result.IsSuccess())
            return result;

        // Get module-specific config
        SCR_ModuleConfig moduleConfig = config.GetModuleConfig("telemetry");

        m_fCaptureIntervalMs = moduleConfig.GetFloatValue("interval_ms", 200);
        m_bVerboseLogging = config.GetBool(SCR_ConfigKeys.LOG_VERBOSE, false);

        Print("[TelemetryModule] Initialized (interval=" + m_fCaptureIntervalMs.ToString() + "ms)", LogLevel.NORMAL);
        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    override bool ValidateTarget(IEntity target)
    {
        if (!target)
            return false;

        // Must have VehicleWheeledSimulation
        VehicleWheeledSimulation sim = VehicleWheeledSimulation.Cast(target.FindComponent(VehicleWheeledSimulation));
        return sim != null;
    }

    //------------------------------------------------------------------------
    override void OnTargetAdded(IEntity target, int targetIndex)
    {
        if (!target)
            return;

        // Ensure array is big enough
        while (m_aTargetStates.Count() <= targetIndex)
        {
            m_aTargetStates.Insert(null);
        }

        // Create state for this target
        m_aTargetStates[targetIndex] = new SCR_TelemetryTargetState(target);

        if (m_aTargetStates[targetIndex].IsValid())
        {
            Print("[TelemetryModule] Added target: " + target.GetName() + " (index " + targetIndex.ToString() + ")", LogLevel.NORMAL);
        }
        else
        {
            Print("[TelemetryModule] Invalid target (no VehicleWheeledSimulation): " + target.GetName(), LogLevel.WARNING);
        }
    }

    //------------------------------------------------------------------------
    override void OnTargetRemoved(IEntity target, int targetIndex)
    {
        if (targetIndex >= 0 && targetIndex < m_aTargetStates.Count())
        {
            m_aTargetStates[targetIndex] = null;
        }
    }

    //------------------------------------------------------------------------
    override string GetCSVHeader()
    {
        return SCR_TelemetryRecord.GetCSVHeader();
    }

    //------------------------------------------------------------------------
    override int GetBinarySchemaVersion()
    {
        return 2;  // v2 includes waypoint context
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Capture(SCR_CaptureContext context, SCR_CaptureBuffer buffer)
    {
        if (!context || !buffer)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "Invalid context or buffer");

        int frameId = context.GetFrameId();
        float timestampMs = context.GetTimestampMs();
        float deltaTimeMs = context.GetDeltaTimeMs();
        float deltaSec = deltaTimeMs / 1000.0;

        int capturedCount = 0;
        int bytesWritten = 0;

        // Capture each target
        array<IEntity> targets = context.GetTargets();
        for (int i = 0; i < targets.Count(); i++)
        {
            IEntity target = targets[i];
            if (!target)
                continue;

            // Get or create state
            if (i >= m_aTargetStates.Count() || !m_aTargetStates[i])
            {
                OnTargetAdded(target, i);
            }

            SCR_TelemetryTargetState state = m_aTargetStates[i];
            if (!state || !state.IsValid())
                continue;

            // Capture telemetry
            SCR_TelemetryRecord record = CaptureTelemetry(state, frameId, timestampMs, i, deltaSec);
            if (record)
            {
                buffer.Write(record, timestampMs);
                capturedCount++;
                bytesWritten += 128;  // Approximate record size

                // Also write directly to serializer for immediate I/O
                // (buffer handles batching internally)
            }
        }

        if (capturedCount > 0)
        {
            RecordCapture(timestampMs);
            return SCR_CaptureResult.Success(bytesWritten, 0);
        }

        return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_INVALID_TARGET, "No valid targets");
    }

    //------------------------------------------------------------------------
    protected SCR_TelemetryRecord CaptureTelemetry(SCR_TelemetryTargetState state, int frameId, float timestampMs, int targetIndex, float deltaSec)
    {
        if (!state || !state.IsValid())
            return null;

        IEntity target = state.m_Target;
        VehicleWheeledSimulation sim = state.m_Simulation;

        // Create record
        SCR_TelemetryRecord record = new SCR_TelemetryRecord(frameId, timestampMs, targetIndex);

        // Get transform
        vector transform[4];
        target.GetWorldTransform(transform);

        vector position = transform[3];
        vector forward = transform[2];
        vector up = transform[1];
        vector right = transform[0];

        // Position
        record.m_fPosX = position[0];
        record.m_fPosY = position[1];
        record.m_fPosZ = position[2];

        // Orientation
        record.m_fFwdX = forward[0];
        record.m_fFwdY = forward[1];
        record.m_fFwdZ = forward[2];

        record.m_fUpX = up[0];
        record.m_fUpY = up[1];
        record.m_fUpZ = up[2];

        record.m_fRightX = right[0];
        record.m_fRightY = right[1];
        record.m_fRightZ = right[2];

        // Vehicle state
        record.m_fSpeedKmh = sim.GetSpeedKmh();
        record.m_fSteering = sim.GetSteering();
        record.m_fThrottle = sim.GetThrottle();
        record.m_fBrake = sim.GetBrake();
        record.m_fClutch = sim.GetClutch();
        record.m_iGear = sim.GetGear();
        record.m_fEngineRPM = sim.EngineGetRPM();
        record.m_bEngineOn = sim.EngineIsOn();
        record.m_bHandbrake = sim.IsHandbrakeOn();

        // Compute distance traveled
        float distance = vector.Distance(position, state.m_vLastPosition);
        state.m_fTotalDistance += distance;
        state.m_vLastPosition = position;

        // Compute acceleration
        if (deltaSec > 0)
        {
            state.m_fAcceleration = (record.m_fSpeedKmh - state.m_fLastSpeed) / deltaSec;
        }
        state.m_fLastSpeed = record.m_fSpeedKmh;

        record.m_fAcceleration = state.m_fAcceleration;
        record.m_fDistanceTotal = state.m_fTotalDistance;

        // Waypoint context
        int wpIdx = state.m_iCurrentWaypointIdx;
        record.m_iWaypointIdx = wpIdx;

        if (wpIdx >= 0)
        {
            if (wpIdx < m_aWaypointTypes.Count())
                record.m_iWaypointType = m_aWaypointTypes[wpIdx];
            if (wpIdx < m_aSpeedLimits.Count())
                record.m_fSpeedLimit = m_aSpeedLimits[wpIdx];
        }

        // Verbose logging
        if (m_bVerboseLogging && frameId % 50 == 0)
        {
            Print("[TelemetryModule] T" + targetIndex.ToString() + ": " +
                  record.m_fSpeedKmh.ToString(5, 1) + " km/h, steer=" +
                  record.m_fSteering.ToString(4, 2), LogLevel.VERBOSE);
        }

        return record;
    }

    //------------------------------------------------------------------------
    // Set waypoint classifications (for all targets)
    void SetWaypointClassifications(array<int> types, array<float> speedLimits, array<string> descriptions)
    {
        m_aWaypointTypes.Clear();
        m_aSpeedLimits.Clear();
        m_aWaypointDescriptions.Clear();

        if (types)
        {
            for (int i = 0; i < types.Count(); i++)
                m_aWaypointTypes.Insert(types[i]);
        }

        if (speedLimits)
        {
            for (int i = 0; i < speedLimits.Count(); i++)
                m_aSpeedLimits.Insert(speedLimits[i]);
        }

        if (descriptions)
        {
            for (int i = 0; i < descriptions.Count(); i++)
                m_aWaypointDescriptions.Insert(descriptions[i]);
        }
    }

    //------------------------------------------------------------------------
    // Set current waypoint for a target
    void SetTargetWaypoint(int targetIndex, int waypointIndex)
    {
        if (targetIndex >= 0 && targetIndex < m_aTargetStates.Count())
        {
            SCR_TelemetryTargetState state = m_aTargetStates[targetIndex];
            if (state)
                state.m_iCurrentWaypointIdx = waypointIndex;
        }
    }

    //------------------------------------------------------------------------
    // Get statistics for a target
    float GetTargetDistance(int targetIndex)
    {
        if (targetIndex >= 0 && targetIndex < m_aTargetStates.Count())
        {
            SCR_TelemetryTargetState state = m_aTargetStates[targetIndex];
            if (state)
                return state.m_fTotalDistance;
        }
        return 0;
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Finalize()
    {
        // Clear states
        m_aTargetStates.Clear();

        Print("[TelemetryModule] Finalized. Total captures: " + m_iTotalCaptureCount.ToString(), LogLevel.NORMAL);

        return super.Finalize();
    }
}
