// ============================================================================
// SCR_RoadModule - Road Context Capture Module
// ============================================================================
//
// Captures road and path context data:
// - Road segment information (if on road)
// - Lane position estimation
// - Road curvature ahead
// - Intersection detection
// - Road surface type
//
// Integrates with SCR_LocalRoadExtractor and SCR_EfficientRoadNet.
//
// ============================================================================

// -----------------------------------------------------------------------------
// Road type enumeration
// -----------------------------------------------------------------------------
enum SCR_RoadType
{
    ROAD_NONE = 0,
    ROAD_HIGHWAY = 1,
    ROAD_MAIN = 2,
    ROAD_SECONDARY = 3,
    ROAD_DIRT = 4,
    ROAD_PATH = 5,
    ROAD_OFFROAD = 255
}

// -----------------------------------------------------------------------------
// Road context record
// -----------------------------------------------------------------------------
class SCR_RoadContextRecord : SCR_CaptureDataRecord
{
    // Road presence
    bool m_bOnRoad;
    int m_iRoadType;
    float m_fRoadWidth;

    // Lane position (estimated)
    float m_fLaneOffset;        // Offset from road center (+ = right, - = left)
    float m_fHeadingOffset;     // Angle difference from road direction

    // Road geometry ahead
    float m_fCurvatureAhead;    // Average curvature over next N meters
    float m_fSlopeAhead;        // Average slope ahead
    int m_iIntersectionsAhead;  // Number of intersections in range

    // Road points (for path planning)
    float m_fRoadPoint1X, m_fRoadPoint1Z;  // Point 10m ahead
    float m_fRoadPoint2X, m_fRoadPoint2Z;  // Point 30m ahead
    float m_fRoadPoint3X, m_fRoadPoint3Z;  // Point 60m ahead

    // Nearest road edge
    float m_fDistToLeftEdge;
    float m_fDistToRightEdge;

    //------------------------------------------------------------------------
    void SCR_RoadContextRecord(int frameId, float timestampMs, int targetIndex)
    {
        SCR_CaptureDataRecord(frameId, timestampMs, "road", targetIndex);

        m_bOnRoad = false;
        m_iRoadType = SCR_RoadType.ROAD_NONE;
        m_fRoadWidth = 0;

        m_fLaneOffset = 0;
        m_fHeadingOffset = 0;

        m_fCurvatureAhead = 0;
        m_fSlopeAhead = 0;
        m_iIntersectionsAhead = 0;

        m_fRoadPoint1X = 0; m_fRoadPoint1Z = 0;
        m_fRoadPoint2X = 0; m_fRoadPoint2Z = 0;
        m_fRoadPoint3X = 0; m_fRoadPoint3Z = 0;

        m_fDistToLeftEdge = -1;
        m_fDistToRightEdge = -1;
    }

    //------------------------------------------------------------------------
    override string ToCSV()
    {
        string row = "";
        row += m_iFrameId.ToString() + ",";
        row += m_fTimestampMs.ToString(12, 1) + ",";
        row += m_iTargetIndex.ToString() + ",";

        int onRoad = 0;
        if (m_bOnRoad)
            onRoad = 1;
        row += onRoad.ToString() + ",";
        row += m_iRoadType.ToString() + ",";
        row += m_fRoadWidth.ToString(6, 2) + ",";

        row += m_fLaneOffset.ToString(6, 3) + ",";
        row += m_fHeadingOffset.ToString(6, 3) + ",";

        row += m_fCurvatureAhead.ToString(8, 5) + ",";
        row += m_fSlopeAhead.ToString(6, 3) + ",";
        row += m_iIntersectionsAhead.ToString() + ",";

        row += m_fRoadPoint1X.ToString(10, 3) + ",";
        row += m_fRoadPoint1Z.ToString(10, 3) + ",";
        row += m_fRoadPoint2X.ToString(10, 3) + ",";
        row += m_fRoadPoint2Z.ToString(10, 3) + ",";
        row += m_fRoadPoint3X.ToString(10, 3) + ",";
        row += m_fRoadPoint3Z.ToString(10, 3) + ",";

        row += m_fDistToLeftEdge.ToString(6, 2) + ",";
        row += m_fDistToRightEdge.ToString(6, 2);

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

        int flags = 0;
        if (m_bOnRoad) flags |= 1;
        file.Write(flags, 1);
        file.Write(m_iRoadType, 1);

        file.Write(m_fRoadWidth, 4);
        file.Write(m_fLaneOffset, 4);
        file.Write(m_fHeadingOffset, 4);
        file.Write(m_fCurvatureAhead, 4);
        file.Write(m_fSlopeAhead, 4);
        file.Write(m_iIntersectionsAhead, 4);

        file.Write(m_fRoadPoint1X, 4);
        file.Write(m_fRoadPoint1Z, 4);
        file.Write(m_fRoadPoint2X, 4);
        file.Write(m_fRoadPoint2Z, 4);
        file.Write(m_fRoadPoint3X, 4);
        file.Write(m_fRoadPoint3Z, 4);

        file.Write(m_fDistToLeftEdge, 4);
        file.Write(m_fDistToRightEdge, 4);
    }

    //------------------------------------------------------------------------
    static override string GetCSVHeader()
    {
        string header = "frame_id,timestamp_ms,target_index,";
        header += "on_road,road_type,road_width,";
        header += "lane_offset,heading_offset,";
        header += "curvature_ahead,slope_ahead,intersections_ahead,";
        header += "road_pt1_x,road_pt1_z,road_pt2_x,road_pt2_z,road_pt3_x,road_pt3_z,";
        header += "dist_left_edge,dist_right_edge";
        return header;
    }
}

// -----------------------------------------------------------------------------
// SCR_RoadModule - Main road capture module
// -----------------------------------------------------------------------------
class SCR_RoadModule : SCR_ICaptureModule
{
    // Configuration
    protected float m_fRoadSearchRadius;
    protected float m_fLookaheadDistance;
    protected int m_iLookaheadSamples;

    // Road detection state
    protected ref array<BaseRoadSegment> m_aCachedRoadSegments;
    protected float m_fLastRoadCacheTime;
    protected float m_fRoadCacheInterval;

    //------------------------------------------------------------------------
    void SCR_RoadModule()
    {
        // Initialize metadata
        m_Metadata = new SCR_ModuleMetadata(
            "road",
            "Road Context",
            "Captures road and lane context information",
            "1.0.0",
            SCR_ModuleCapability.CAP_REAL_TIME | SCR_ModuleCapability.CAP_MULTI_TARGET,
            SCR_CaptureFormat.FORMAT_CSV | SCR_CaptureFormat.FORMAT_BINARY,
            200,    // 5 Hz default
            25      // Priority (after depth, before scene)
        );

        // Default configuration
        m_fRoadSearchRadius = 50.0;
        m_fLookaheadDistance = 100.0;
        m_iLookaheadSamples = 10;

        m_aCachedRoadSegments = new array<BaseRoadSegment>();
        m_fLastRoadCacheTime = 0;
        m_fRoadCacheInterval = 1000;  // Cache roads every 1 second
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Initialize(SCR_CaptureConfig config)
    {
        SCR_CaptureResult result = super.Initialize(config);
        if (!result.IsSuccess())
            return result;

        // Get module-specific config
        SCR_ModuleConfig moduleConfig = config.GetModuleConfig("road");

        m_fRoadSearchRadius = moduleConfig.GetFloatValue("search_radius", m_fRoadSearchRadius);
        m_fLookaheadDistance = moduleConfig.GetFloatValue("lookahead_distance", m_fLookaheadDistance);
        m_iLookaheadSamples = moduleConfig.GetIntValue("lookahead_samples", m_iLookaheadSamples);
        m_fRoadCacheInterval = moduleConfig.GetFloatValue("cache_interval", m_fRoadCacheInterval);

        Print("[RoadModule] Initialized (radius=" + m_fRoadSearchRadius.ToString() + "m, lookahead=" + m_fLookaheadDistance.ToString() + "m)", LogLevel.NORMAL);
        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    override string GetCSVHeader()
    {
        return SCR_RoadContextRecord.GetCSVHeader();
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

            // Capture road context
            SCR_RoadContextRecord record = CaptureRoadContext(target, frameId, timestampMs, t);
            if (record)
            {
                buffer.Write(record, timestampMs);
                bytesWritten += 80;
                capturedCount++;
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
    protected SCR_RoadContextRecord CaptureRoadContext(IEntity target, int frameId, float timestampMs, int targetIndex)
    {
        SCR_RoadContextRecord record = new SCR_RoadContextRecord(frameId, timestampMs, targetIndex);

        // Get target transform
        vector transform[4];
        target.GetWorldTransform(transform);
        vector position = transform[3];
        vector forward = transform[2];

        // Update road cache if needed
        UpdateRoadCache(position, timestampMs);

        // Find nearest road segment
        BaseRoadSegment nearestSegment = FindNearestRoadSegment(position);

        if (nearestSegment)
        {
            record.m_bOnRoad = true;
            record.m_fRoadWidth = GetRoadWidth(nearestSegment);
            record.m_iRoadType = ClassifyRoadType(nearestSegment);

            // Calculate lane offset and heading
            CalculateLanePosition(position, forward, nearestSegment, record);

            // Calculate road points ahead
            CalculateRoadPointsAhead(position, forward, record);

            // Calculate edge distances
            CalculateEdgeDistances(position, nearestSegment, record);

            // Calculate curvature ahead
            CalculateCurvatureAhead(position, forward, record);
        }
        else
        {
            record.m_bOnRoad = false;
            record.m_iRoadType = SCR_RoadType.ROAD_OFFROAD;
        }

        return record;
    }

    //------------------------------------------------------------------------
    protected void UpdateRoadCache(vector position, float timestampMs)
    {
        if ((timestampMs - m_fLastRoadCacheTime) < m_fRoadCacheInterval)
            return;

        m_aCachedRoadSegments.Clear();

        // Get road network manager
        RoadNetworkManager roadManager = GetGame().GetRoadNetworkManager();
        if (!roadManager)
            return;

        // Query road segments in radius
        // Note: Actual API may vary based on Enfusion version
        // This is a simplified implementation
        roadManager.GetRoadSegmentsInArea(position, m_fRoadSearchRadius, m_aCachedRoadSegments);

        m_fLastRoadCacheTime = timestampMs;
    }

    //------------------------------------------------------------------------
    protected BaseRoadSegment FindNearestRoadSegment(vector position)
    {
        BaseRoadSegment nearest = null;
        float nearestDist = m_fRoadSearchRadius;

        for (int i = 0; i < m_aCachedRoadSegments.Count(); i++)
        {
            BaseRoadSegment segment = m_aCachedRoadSegments[i];
            if (!segment)
                continue;

            float dist = GetDistanceToRoadSegment(position, segment);
            if (dist < nearestDist)
            {
                nearestDist = dist;
                nearest = segment;
            }
        }

        return nearest;
    }

    //------------------------------------------------------------------------
    protected float GetDistanceToRoadSegment(vector position, BaseRoadSegment segment)
    {
        if (!segment)
            return m_fRoadSearchRadius;

        // Get closest point on road to position
        vector closestPoint;
        segment.GetClosestPointOnRoad(position, closestPoint);

        return vector.Distance(position, closestPoint);
    }

    //------------------------------------------------------------------------
    protected float GetRoadWidth(BaseRoadSegment segment)
    {
        if (!segment)
            return 0;

        // Get road width from segment
        // Default to reasonable values based on road type
        return segment.GetRoadWidth();
    }

    //------------------------------------------------------------------------
    protected int ClassifyRoadType(BaseRoadSegment segment)
    {
        if (!segment)
            return SCR_RoadType.ROAD_NONE;

        // Classify based on road properties
        float width = segment.GetRoadWidth();
        int flags = segment.GetRoadFlags();

        // Simple classification based on width
        if (width > 12.0)
            return SCR_RoadType.ROAD_HIGHWAY;
        if (width > 8.0)
            return SCR_RoadType.ROAD_MAIN;
        if (width > 5.0)
            return SCR_RoadType.ROAD_SECONDARY;
        if (width > 3.0)
            return SCR_RoadType.ROAD_DIRT;

        return SCR_RoadType.ROAD_PATH;
    }

    //------------------------------------------------------------------------
    protected void CalculateLanePosition(vector position, vector forward, BaseRoadSegment segment, SCR_RoadContextRecord record)
    {
        if (!segment)
            return;

        // Get road center line and direction at this point
        vector roadCenter;
        vector roadDirection;

        segment.GetClosestPointOnRoad(position, roadCenter);
        segment.GetRoadDirectionAt(roadCenter, roadDirection);

        // Calculate lateral offset from center
        vector toVehicle = position - roadCenter;
        vector roadRight = roadDirection * "0 1 0";  // Cross with up to get right

        record.m_fLaneOffset = vector.Dot(toVehicle, roadRight);

        // Calculate heading offset
        float dot = vector.Dot(forward, roadDirection);
        record.m_fHeadingOffset = Math.Acos(Math.Clamp(dot, -1.0, 1.0)) * Math.RAD2DEG;

        // Determine if heading is reversed
        vector cross = forward * roadDirection;
        if (cross[1] < 0)
            record.m_fHeadingOffset = -record.m_fHeadingOffset;
    }

    //------------------------------------------------------------------------
    protected void CalculateRoadPointsAhead(vector position, vector forward, SCR_RoadContextRecord record)
    {
        // Sample road positions ahead
        float[] distances = {10.0, 30.0, 60.0};

        for (int i = 0; i < 3; i++)
        {
            vector ahead = position + forward * distances[i];

            // Find road point nearest to projected position
            BaseRoadSegment segment = FindNearestRoadSegment(ahead);
            if (segment)
            {
                vector roadPoint;
                segment.GetClosestPointOnRoad(ahead, roadPoint);

                // Store relative to current position
                vector relative = roadPoint - position;

                switch (i)
                {
                    case 0:
                        record.m_fRoadPoint1X = relative[0];
                        record.m_fRoadPoint1Z = relative[2];
                        break;
                    case 1:
                        record.m_fRoadPoint2X = relative[0];
                        record.m_fRoadPoint2Z = relative[2];
                        break;
                    case 2:
                        record.m_fRoadPoint3X = relative[0];
                        record.m_fRoadPoint3Z = relative[2];
                        break;
                }
            }
        }
    }

    //------------------------------------------------------------------------
    protected void CalculateEdgeDistances(vector position, BaseRoadSegment segment, SCR_RoadContextRecord record)
    {
        if (!segment)
            return;

        vector roadCenter;
        segment.GetClosestPointOnRoad(position, roadCenter);

        vector roadDirection;
        segment.GetRoadDirectionAt(roadCenter, roadDirection);

        float halfWidth = segment.GetRoadWidth() / 2.0;

        // Calculate right vector of road
        vector roadRight = roadDirection * "0 1 0";
        roadRight.Normalize();

        // Calculate distances to edges
        vector leftEdge = roadCenter - roadRight * halfWidth;
        vector rightEdge = roadCenter + roadRight * halfWidth;

        // Project position onto road perpendicular
        vector toVehicle = position - roadCenter;
        float lateralPos = vector.Dot(toVehicle, roadRight);

        record.m_fDistToLeftEdge = halfWidth + lateralPos;
        record.m_fDistToRightEdge = halfWidth - lateralPos;
    }

    //------------------------------------------------------------------------
    protected void CalculateCurvatureAhead(vector position, vector forward, SCR_RoadContextRecord record)
    {
        float totalCurvature = 0;
        int samples = 0;

        float sampleStep = m_fLookaheadDistance / m_iLookaheadSamples;
        vector lastDirection = forward;

        for (int i = 1; i <= m_iLookaheadSamples; i++)
        {
            float dist = i * sampleStep;
            vector samplePos = position + forward * dist;

            BaseRoadSegment segment = FindNearestRoadSegment(samplePos);
            if (segment)
            {
                vector roadPoint;
                vector roadDir;
                segment.GetClosestPointOnRoad(samplePos, roadPoint);
                segment.GetRoadDirectionAt(roadPoint, roadDir);

                // Calculate angle change
                float dot = vector.Dot(lastDirection, roadDir);
                float angle = Math.Acos(Math.Clamp(dot, -1.0, 1.0));

                // Curvature = angle change / distance
                totalCurvature += angle / sampleStep;
                samples++;

                lastDirection = roadDir;
            }
        }

        if (samples > 0)
        {
            record.m_fCurvatureAhead = totalCurvature / samples;
        }
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Finalize()
    {
        m_aCachedRoadSegments.Clear();

        Print("[RoadModule] Finalized. Total captures: " + m_iTotalCaptureCount.ToString(), LogLevel.NORMAL);
        return super.Finalize();
    }
}
