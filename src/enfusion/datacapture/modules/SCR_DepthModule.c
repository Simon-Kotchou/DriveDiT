// ============================================================================
// SCR_DepthModule - Depth/Distance Capture Module
// ============================================================================
//
// Captures depth and distance data using raycasting:
// - Forward raycast distances (obstacle detection)
// - Multi-ray fan pattern for field-of-view depth
// - Ground distance (terrain height)
// - Optional depth buffer export (when available)
//
// OUTPUT FORMAT (CSV):
//   frame_id, timestamp_ms, target_index, ray_index, distance, hit_type, ...
//
// ============================================================================

// -----------------------------------------------------------------------------
// Depth ray hit types
// -----------------------------------------------------------------------------
enum SCR_DepthHitType
{
    HIT_NONE = 0,           // Ray didn't hit anything
    HIT_TERRAIN = 1,        // Hit terrain/ground
    HIT_BUILDING = 2,       // Hit building/structure
    HIT_VEHICLE = 3,        // Hit vehicle
    HIT_VEGETATION = 4,     // Hit tree/vegetation
    HIT_PROP = 5,           // Hit static prop
    HIT_CHARACTER = 6,      // Hit character/AI
    HIT_WATER = 7,          // Hit water surface
    HIT_UNKNOWN = 255
}

// -----------------------------------------------------------------------------
// Depth data record (per ray)
// -----------------------------------------------------------------------------
class SCR_DepthRayRecord : SCR_CaptureDataRecord
{
    int m_iRayIndex;
    float m_fDistance;
    int m_iHitType;
    float m_fHitPosX, m_fHitPosY, m_fHitPosZ;
    float m_fRayDirX, m_fRayDirY, m_fRayDirZ;
    float m_fAngleH;    // Horizontal angle offset from forward
    float m_fAngleV;    // Vertical angle offset from forward

    //------------------------------------------------------------------------
    void SCR_DepthRayRecord(int frameId, float timestampMs, int targetIndex, int rayIndex)
    {
        SCR_CaptureDataRecord(frameId, timestampMs, "depth", targetIndex);
        m_iRayIndex = rayIndex;
        m_fDistance = -1;
        m_iHitType = SCR_DepthHitType.HIT_NONE;
        m_fHitPosX = 0; m_fHitPosY = 0; m_fHitPosZ = 0;
        m_fRayDirX = 0; m_fRayDirY = 0; m_fRayDirZ = 1;
        m_fAngleH = 0;
        m_fAngleV = 0;
    }

    //------------------------------------------------------------------------
    override string ToCSV()
    {
        string row = "";
        row += m_iFrameId.ToString() + ",";
        row += m_fTimestampMs.ToString(12, 1) + ",";
        row += m_iTargetIndex.ToString() + ",";
        row += m_iRayIndex.ToString() + ",";
        row += m_fDistance.ToString(8, 3) + ",";
        row += m_iHitType.ToString() + ",";
        row += m_fHitPosX.ToString(10, 4) + ",";
        row += m_fHitPosY.ToString(10, 4) + ",";
        row += m_fHitPosZ.ToString(10, 4) + ",";
        row += m_fRayDirX.ToString(8, 6) + ",";
        row += m_fRayDirY.ToString(8, 6) + ",";
        row += m_fRayDirZ.ToString(8, 6) + ",";
        row += m_fAngleH.ToString(6, 2) + ",";
        row += m_fAngleV.ToString(6, 2);
        return row;
    }

    //------------------------------------------------------------------------
    override void ToBinary(FileHandle file)
    {
        if (!file)
            return;

        file.Write(m_iFrameId, 4);
        file.Write(m_fTimestampMs, 4);
        file.Write(m_iTargetIndex, 2);
        file.Write(m_iRayIndex, 2);
        file.Write(m_fDistance, 4);
        file.Write(m_iHitType, 1);
        // Pad to 4-byte alignment
        file.Write(0, 3);
        file.Write(m_fHitPosX, 4);
        file.Write(m_fHitPosY, 4);
        file.Write(m_fHitPosZ, 4);
    }

    //------------------------------------------------------------------------
    static override string GetCSVHeader()
    {
        return "frame_id,timestamp_ms,target_index,ray_index,distance,hit_type,hit_x,hit_y,hit_z,dir_x,dir_y,dir_z,angle_h,angle_v";
    }
}

// -----------------------------------------------------------------------------
// Aggregated depth summary record (one per target per frame)
// -----------------------------------------------------------------------------
class SCR_DepthSummaryRecord : SCR_CaptureDataRecord
{
    float m_fMinDistance;
    float m_fMaxDistance;
    float m_fMeanDistance;
    float m_fForwardDistance;
    float m_fGroundDistance;
    int m_iRayCount;
    int m_iHitCount;

    //------------------------------------------------------------------------
    void SCR_DepthSummaryRecord(int frameId, float timestampMs, int targetIndex)
    {
        SCR_CaptureDataRecord(frameId, timestampMs, "depth_summary", targetIndex);
        m_fMinDistance = -1;
        m_fMaxDistance = -1;
        m_fMeanDistance = -1;
        m_fForwardDistance = -1;
        m_fGroundDistance = -1;
        m_iRayCount = 0;
        m_iHitCount = 0;
    }

    //------------------------------------------------------------------------
    override string ToCSV()
    {
        string row = "";
        row += m_iFrameId.ToString() + ",";
        row += m_fTimestampMs.ToString(12, 1) + ",";
        row += m_iTargetIndex.ToString() + ",";
        row += m_fMinDistance.ToString(8, 3) + ",";
        row += m_fMaxDistance.ToString(8, 3) + ",";
        row += m_fMeanDistance.ToString(8, 3) + ",";
        row += m_fForwardDistance.ToString(8, 3) + ",";
        row += m_fGroundDistance.ToString(8, 3) + ",";
        row += m_iRayCount.ToString() + ",";
        row += m_iHitCount.ToString();
        return row;
    }

    //------------------------------------------------------------------------
    static override string GetCSVHeader()
    {
        return "frame_id,timestamp_ms,target_index,min_dist,max_dist,mean_dist,forward_dist,ground_dist,ray_count,hit_count";
    }
}

// -----------------------------------------------------------------------------
// SCR_DepthModule - Main depth capture module
// -----------------------------------------------------------------------------
class SCR_DepthModule : SCR_ICaptureModule
{
    // Ray pattern configuration
    protected int m_iHorizontalRays;
    protected int m_iVerticalRays;
    protected float m_fHorizontalFOV;
    protected float m_fVerticalFOV;
    protected float m_fMaxRayDistance;

    // Ray origin offset (from entity center)
    protected vector m_vRayOriginOffset;

    // Output mode
    protected bool m_bOutputPerRay;
    protected bool m_bOutputSummary;

    // Trace flags
    protected TraceFlags m_iTraceFlags;

    // Pre-computed ray directions (relative to forward)
    protected ref array<vector> m_aRayDirections;
    protected ref array<float> m_aRayAnglesH;
    protected ref array<float> m_aRayAnglesV;

    //------------------------------------------------------------------------
    void SCR_DepthModule()
    {
        // Initialize metadata
        m_Metadata = new SCR_ModuleMetadata(
            "depth",
            "Depth Sensing",
            "Captures raycast-based depth and distance measurements",
            "1.0.0",
            SCR_ModuleCapability.CAP_REAL_TIME | SCR_ModuleCapability.CAP_MULTI_TARGET,
            SCR_CaptureFormat.FORMAT_CSV | SCR_CaptureFormat.FORMAT_BINARY,
            200,    // 5 Hz default
            20      // Priority (after telemetry)
        );

        // Default configuration
        m_iHorizontalRays = 11;     // -50 to +50 degrees in 10-degree steps
        m_iVerticalRays = 3;        // -10, 0, +10 degrees
        m_fHorizontalFOV = 100.0;   // Total horizontal FOV in degrees
        m_fVerticalFOV = 20.0;      // Total vertical FOV in degrees
        m_fMaxRayDistance = 200.0;  // Max raycast distance

        m_vRayOriginOffset = "0 1.5 0.5";  // Eye-level, slightly forward

        m_bOutputPerRay = false;
        m_bOutputSummary = true;

        m_aRayDirections = new array<vector>();
        m_aRayAnglesH = new array<float>();
        m_aRayAnglesV = new array<float>();

        // Trace flags - hit static and dynamic geometry
        m_iTraceFlags = TraceFlags.WORLD | TraceFlags.ENTS;
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Initialize(SCR_CaptureConfig config)
    {
        SCR_CaptureResult result = super.Initialize(config);
        if (!result.IsSuccess())
            return result;

        // Get module-specific config
        SCR_ModuleConfig moduleConfig = config.GetModuleConfig("depth");

        m_iHorizontalRays = moduleConfig.GetIntValue("horizontal_rays", m_iHorizontalRays);
        m_iVerticalRays = moduleConfig.GetIntValue("vertical_rays", m_iVerticalRays);
        m_fHorizontalFOV = moduleConfig.GetFloatValue("horizontal_fov", m_fHorizontalFOV);
        m_fVerticalFOV = moduleConfig.GetFloatValue("vertical_fov", m_fVerticalFOV);
        m_fMaxRayDistance = moduleConfig.GetFloatValue("max_distance", m_fMaxRayDistance);
        m_bOutputPerRay = moduleConfig.GetBoolValue("output_per_ray", m_bOutputPerRay);
        m_bOutputSummary = moduleConfig.GetBoolValue("output_summary", m_bOutputSummary);

        // Pre-compute ray pattern
        ComputeRayPattern();

        Print("[DepthModule] Initialized (" + m_aRayDirections.Count().ToString() + " rays, " +
              m_fMaxRayDistance.ToString(5, 1) + "m max)", LogLevel.NORMAL);

        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    protected void ComputeRayPattern()
    {
        m_aRayDirections.Clear();
        m_aRayAnglesH.Clear();
        m_aRayAnglesV.Clear();

        float hStep = m_fHorizontalFOV / Math.Max(1, m_iHorizontalRays - 1);
        float vStep = m_fVerticalFOV / Math.Max(1, m_iVerticalRays - 1);

        float hStart = -m_fHorizontalFOV / 2.0;
        float vStart = -m_fVerticalFOV / 2.0;

        for (int v = 0; v < m_iVerticalRays; v++)
        {
            float angleV = vStart + v * vStep;

            for (int h = 0; h < m_iHorizontalRays; h++)
            {
                float angleH = hStart + h * hStep;

                // Convert to direction vector (relative to forward = 0,0,1)
                float radH = angleH * Math.DEG2RAD;
                float radV = angleV * Math.DEG2RAD;

                float x = Math.Sin(radH) * Math.Cos(radV);
                float y = Math.Sin(radV);
                float z = Math.Cos(radH) * Math.Cos(radV);

                vector dir = Vector(x, y, z);
                dir.Normalize();

                m_aRayDirections.Insert(dir);
                m_aRayAnglesH.Insert(angleH);
                m_aRayAnglesV.Insert(angleV);
            }
        }
    }

    //------------------------------------------------------------------------
    override string GetCSVHeader()
    {
        if (m_bOutputPerRay)
            return SCR_DepthRayRecord.GetCSVHeader();
        return SCR_DepthSummaryRecord.GetCSVHeader();
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Capture(SCR_CaptureContext context, SCR_CaptureBuffer buffer)
    {
        if (!context || !buffer)
            return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_CONFIG_INVALID, "Invalid context or buffer");

        int frameId = context.GetFrameId();
        float timestampMs = context.GetTimestampMs();

        int bytesWritten = 0;
        int capturedCount = 0;

        // Capture for each target
        array<IEntity> targets = context.GetTargets();
        for (int t = 0; t < targets.Count(); t++)
        {
            IEntity target = targets[t];
            if (!target)
                continue;

            // Get target transform
            vector transform[4];
            target.GetWorldTransform(transform);

            vector origin = transform[3] + m_vRayOriginOffset;
            vector forward = transform[2];
            vector up = transform[1];
            vector right = transform[0];

            // Cast all rays
            ref array<ref SCR_DepthRayRecord> rayRecords = new array<ref SCR_DepthRayRecord>();
            float sumDistance = 0;
            float minDistance = m_fMaxRayDistance;
            float maxDistance = 0;
            int hitCount = 0;
            float forwardDistance = -1;
            float groundDistance = -1;

            for (int r = 0; r < m_aRayDirections.Count(); r++)
            {
                vector localDir = m_aRayDirections[r];

                // Transform to world space
                vector worldDir = right * localDir[0] + up * localDir[1] + forward * localDir[2];
                worldDir.Normalize();

                // Perform raycast
                float distance = -1;
                int hitType = SCR_DepthHitType.HIT_NONE;
                vector hitPos = vector.Zero;

                TraceParam trace = new TraceParam();
                trace.Start = origin;
                trace.End = origin + worldDir * m_fMaxRayDistance;
                trace.Flags = m_iTraceFlags;
                trace.LayerMask = TRACE_LAYER_CAMERA;

                float fraction = GetGame().GetWorld().TraceMove(trace, null);
                if (fraction < 1.0)
                {
                    distance = fraction * m_fMaxRayDistance;
                    hitPos = trace.Start + worldDir * distance;
                    hitType = ClassifyHit(trace);
                    hitCount++;

                    sumDistance += distance;
                    if (distance < minDistance)
                        minDistance = distance;
                    if (distance > maxDistance)
                        maxDistance = distance;
                }

                // Check for center ray (forward)
                if (r == (m_aRayDirections.Count() / 2))
                {
                    forwardDistance = distance;
                }

                // Output per-ray record
                if (m_bOutputPerRay)
                {
                    SCR_DepthRayRecord record = new SCR_DepthRayRecord(frameId, timestampMs, t, r);
                    record.m_fDistance = distance;
                    record.m_iHitType = hitType;
                    record.m_fHitPosX = hitPos[0];
                    record.m_fHitPosY = hitPos[1];
                    record.m_fHitPosZ = hitPos[2];
                    record.m_fRayDirX = worldDir[0];
                    record.m_fRayDirY = worldDir[1];
                    record.m_fRayDirZ = worldDir[2];
                    record.m_fAngleH = m_aRayAnglesH[r];
                    record.m_fAngleV = m_aRayAnglesV[r];

                    buffer.Write(record, timestampMs);
                    bytesWritten += 64;
                }
            }

            // Ground ray (straight down)
            {
                TraceParam groundTrace = new TraceParam();
                groundTrace.Start = origin;
                groundTrace.End = origin - "0 100 0";  // 100m down
                groundTrace.Flags = m_iTraceFlags;
                groundTrace.LayerMask = TRACE_LAYER_CAMERA;

                float groundFraction = GetGame().GetWorld().TraceMove(groundTrace, null);
                if (groundFraction < 1.0)
                {
                    groundDistance = groundFraction * 100.0;
                }
            }

            // Output summary record
            if (m_bOutputSummary)
            {
                SCR_DepthSummaryRecord summary = new SCR_DepthSummaryRecord(frameId, timestampMs, t);
                summary.m_iRayCount = m_aRayDirections.Count();
                summary.m_iHitCount = hitCount;

                if (hitCount > 0)
                {
                    summary.m_fMinDistance = minDistance;
                    summary.m_fMaxDistance = maxDistance;
                    summary.m_fMeanDistance = sumDistance / hitCount;
                }
                else
                {
                    summary.m_fMinDistance = -1;
                    summary.m_fMaxDistance = -1;
                    summary.m_fMeanDistance = -1;
                }

                summary.m_fForwardDistance = forwardDistance;
                summary.m_fGroundDistance = groundDistance;

                buffer.Write(summary, timestampMs);
                bytesWritten += 48;
            }

            capturedCount++;
        }

        if (capturedCount > 0)
        {
            RecordCapture(timestampMs);
            return SCR_CaptureResult.Success(bytesWritten, 0);
        }

        return SCR_CaptureResult.Failure(SCR_CaptureError.CAPTURE_ERROR_INVALID_TARGET, "No valid targets");
    }

    //------------------------------------------------------------------------
    protected int ClassifyHit(TraceParam trace)
    {
        // Attempt to classify what was hit
        // This is simplified - real implementation would inspect trace results
        // and classify based on entity type, material, etc.

        if (trace.TraceEnt)
        {
            // Check if it's a vehicle
            VehicleWheeledSimulation vehicleSim = VehicleWheeledSimulation.Cast(trace.TraceEnt.FindComponent(VehicleWheeledSimulation));
            if (vehicleSim)
                return SCR_DepthHitType.HIT_VEHICLE;

            // Check entity type name for hints
            string className = trace.TraceEnt.ClassName();
            if (className.Contains("Building") || className.Contains("House"))
                return SCR_DepthHitType.HIT_BUILDING;
            if (className.Contains("Tree") || className.Contains("Bush"))
                return SCR_DepthHitType.HIT_VEGETATION;
            if (className.Contains("Character") || className.Contains("Soldier"))
                return SCR_DepthHitType.HIT_CHARACTER;

            return SCR_DepthHitType.HIT_PROP;
        }

        // No entity hit - likely terrain
        return SCR_DepthHitType.HIT_TERRAIN;
    }

    //------------------------------------------------------------------------
    // Configure ray pattern
    void SetRayPattern(int horizontalRays, int verticalRays, float horizontalFOV, float verticalFOV)
    {
        m_iHorizontalRays = horizontalRays;
        m_iVerticalRays = verticalRays;
        m_fHorizontalFOV = horizontalFOV;
        m_fVerticalFOV = verticalFOV;
        ComputeRayPattern();
    }

    void SetMaxDistance(float maxDistance)
    {
        m_fMaxRayDistance = maxDistance;
    }

    void SetOutputMode(bool perRay, bool summary)
    {
        m_bOutputPerRay = perRay;
        m_bOutputSummary = summary;
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Finalize()
    {
        m_aRayDirections.Clear();
        m_aRayAnglesH.Clear();
        m_aRayAnglesV.Clear();

        Print("[DepthModule] Finalized. Total captures: " + m_iTotalCaptureCount.ToString(), LogLevel.NORMAL);

        return super.Finalize();
    }
}
