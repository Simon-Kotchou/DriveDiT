// ============================================================================
// SCR_SceneModule - Scene Context Capture Module
// ============================================================================
//
// Captures environmental and scene context data:
// - Weather conditions (sky state, precipitation, fog)
// - Time of day (sun angle, lighting conditions)
// - Nearby entities (vehicles, characters, objects)
// - Road/terrain context
// - Visibility conditions
//
// OUTPUT FORMAT (CSV):
//   frame_id, timestamp_ms, time_of_day, weather_state, fog_density, ...
//
// ============================================================================

// -----------------------------------------------------------------------------
// Weather state enumeration
// -----------------------------------------------------------------------------
enum SCR_WeatherState
{
    WEATHER_CLEAR = 0,
    WEATHER_CLOUDY = 1,
    WEATHER_OVERCAST = 2,
    WEATHER_RAINY = 3,
    WEATHER_STORMY = 4,
    WEATHER_FOGGY = 5,
    WEATHER_SNOWY = 6,
    WEATHER_UNKNOWN = 255
}

// -----------------------------------------------------------------------------
// Nearby entity record
// -----------------------------------------------------------------------------
class SCR_NearbyEntityRecord : SCR_CaptureDataRecord
{
    int m_iEntityIndex;
    string m_sEntityType;
    float m_fRelativeX, m_fRelativeY, m_fRelativeZ;
    float m_fDistance;
    float m_fBearing;
    float m_fVelocity;

    //------------------------------------------------------------------------
    void SCR_NearbyEntityRecord(int frameId, float timestampMs, int targetIndex, int entityIndex)
    {
        SCR_CaptureDataRecord(frameId, timestampMs, "scene_entity", targetIndex);
        m_iEntityIndex = entityIndex;
        m_sEntityType = "";
        m_fRelativeX = 0; m_fRelativeY = 0; m_fRelativeZ = 0;
        m_fDistance = 0;
        m_fBearing = 0;
        m_fVelocity = 0;
    }

    //------------------------------------------------------------------------
    override string ToCSV()
    {
        string row = "";
        row += m_iFrameId.ToString() + ",";
        row += m_fTimestampMs.ToString(12, 1) + ",";
        row += m_iTargetIndex.ToString() + ",";
        row += m_iEntityIndex.ToString() + ",";
        row += m_sEntityType + ",";
        row += m_fRelativeX.ToString(8, 3) + ",";
        row += m_fRelativeY.ToString(8, 3) + ",";
        row += m_fRelativeZ.ToString(8, 3) + ",";
        row += m_fDistance.ToString(8, 3) + ",";
        row += m_fBearing.ToString(6, 2) + ",";
        row += m_fVelocity.ToString(8, 3);
        return row;
    }

    //------------------------------------------------------------------------
    static override string GetCSVHeader()
    {
        return "frame_id,timestamp_ms,target_index,entity_index,entity_type,rel_x,rel_y,rel_z,distance,bearing,velocity";
    }
}

// -----------------------------------------------------------------------------
// Scene context summary record
// -----------------------------------------------------------------------------
class SCR_SceneContextRecord : SCR_CaptureDataRecord
{
    // Time
    float m_fTimeOfDay;
    float m_fSunAngle;

    // Weather
    int m_iWeatherState;
    float m_fFogDensity;
    float m_fRainIntensity;
    float m_fCloudCover;
    float m_fWindSpeed;
    float m_fWindDirection;

    // Visibility
    float m_fVisibilityRange;
    float m_fAmbientLight;

    // Nearby counts
    int m_iNearbyVehicles;
    int m_iNearbyCharacters;
    int m_iNearbyObjects;
    float m_fNearestObstacle;

    // Terrain
    string m_sTerrainType;
    float m_fTerrainSlope;

    //------------------------------------------------------------------------
    void SCR_SceneContextRecord(int frameId, float timestampMs, int targetIndex)
    {
        SCR_CaptureDataRecord(frameId, timestampMs, "scene", targetIndex);

        m_fTimeOfDay = 12.0;
        m_fSunAngle = 45.0;

        m_iWeatherState = SCR_WeatherState.WEATHER_CLEAR;
        m_fFogDensity = 0;
        m_fRainIntensity = 0;
        m_fCloudCover = 0;
        m_fWindSpeed = 0;
        m_fWindDirection = 0;

        m_fVisibilityRange = 10000;
        m_fAmbientLight = 1.0;

        m_iNearbyVehicles = 0;
        m_iNearbyCharacters = 0;
        m_iNearbyObjects = 0;
        m_fNearestObstacle = -1;

        m_sTerrainType = "unknown";
        m_fTerrainSlope = 0;
    }

    //------------------------------------------------------------------------
    override string ToCSV()
    {
        string row = "";
        row += m_iFrameId.ToString() + ",";
        row += m_fTimestampMs.ToString(12, 1) + ",";
        row += m_iTargetIndex.ToString() + ",";

        // Time
        row += m_fTimeOfDay.ToString(5, 2) + ",";
        row += m_fSunAngle.ToString(6, 2) + ",";

        // Weather
        row += m_iWeatherState.ToString() + ",";
        row += m_fFogDensity.ToString(6, 4) + ",";
        row += m_fRainIntensity.ToString(6, 4) + ",";
        row += m_fCloudCover.ToString(6, 4) + ",";
        row += m_fWindSpeed.ToString(6, 2) + ",";
        row += m_fWindDirection.ToString(6, 2) + ",";

        // Visibility
        row += m_fVisibilityRange.ToString(8, 1) + ",";
        row += m_fAmbientLight.ToString(6, 4) + ",";

        // Nearby
        row += m_iNearbyVehicles.ToString() + ",";
        row += m_iNearbyCharacters.ToString() + ",";
        row += m_iNearbyObjects.ToString() + ",";
        row += m_fNearestObstacle.ToString(8, 3) + ",";

        // Terrain
        row += m_sTerrainType + ",";
        row += m_fTerrainSlope.ToString(6, 2);

        return row;
    }

    //------------------------------------------------------------------------
    override void ToBinary(FileHandle file)
    {
        if (!file)
            return;

        file.Write(m_iFrameId, 4);
        file.Write(m_fTimestampMs, 4);
        file.Write(m_iTargetIndex, 4);
        file.Write(m_fTimeOfDay, 4);
        file.Write(m_fSunAngle, 4);
        file.Write(m_iWeatherState, 4);
        file.Write(m_fFogDensity, 4);
        file.Write(m_fRainIntensity, 4);
        file.Write(m_fCloudCover, 4);
        file.Write(m_fWindSpeed, 4);
        file.Write(m_fWindDirection, 4);
        file.Write(m_fVisibilityRange, 4);
        file.Write(m_fAmbientLight, 4);
        file.Write(m_iNearbyVehicles, 4);
        file.Write(m_iNearbyCharacters, 4);
        file.Write(m_iNearbyObjects, 4);
        file.Write(m_fNearestObstacle, 4);
        file.Write(m_fTerrainSlope, 4);
    }

    //------------------------------------------------------------------------
    static override string GetCSVHeader()
    {
        string header = "frame_id,timestamp_ms,target_index,";
        header += "time_of_day,sun_angle,";
        header += "weather_state,fog_density,rain_intensity,cloud_cover,wind_speed,wind_direction,";
        header += "visibility_range,ambient_light,";
        header += "nearby_vehicles,nearby_characters,nearby_objects,nearest_obstacle,";
        header += "terrain_type,terrain_slope";
        return header;
    }
}

// -----------------------------------------------------------------------------
// SCR_SceneModule - Main scene capture module
// -----------------------------------------------------------------------------
class SCR_SceneModule : SCR_ICaptureModule
{
    // Configuration
    protected float m_fNearbySearchRadius;
    protected int m_iMaxNearbyEntities;
    protected bool m_bCaptureNearbyEntities;
    protected bool m_bCaptureTerrainInfo;

    // Cached references
    protected TimeAndWeatherManagerEntity m_WeatherManager;
    protected bool m_bWeatherManagerCached;

    //------------------------------------------------------------------------
    void SCR_SceneModule()
    {
        // Initialize metadata
        m_Metadata = new SCR_ModuleMetadata(
            "scene",
            "Scene Context",
            "Captures environmental conditions and nearby entities",
            "1.0.0",
            SCR_ModuleCapability.CAP_REAL_TIME | SCR_ModuleCapability.CAP_MULTI_TARGET,
            SCR_CaptureFormat.FORMAT_CSV | SCR_CaptureFormat.FORMAT_BINARY,
            500,    // 2 Hz default (scene changes slowly)
            30      // Priority (after depth)
        );

        // Default configuration
        m_fNearbySearchRadius = 100.0;
        m_iMaxNearbyEntities = 20;
        m_bCaptureNearbyEntities = true;
        m_bCaptureTerrainInfo = true;

        m_bWeatherManagerCached = false;
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Initialize(SCR_CaptureConfig config)
    {
        SCR_CaptureResult result = super.Initialize(config);
        if (!result.IsSuccess())
            return result;

        // Get module-specific config
        SCR_ModuleConfig moduleConfig = config.GetModuleConfig("scene");

        m_fNearbySearchRadius = moduleConfig.GetFloatValue("search_radius", m_fNearbySearchRadius);
        m_iMaxNearbyEntities = moduleConfig.GetIntValue("max_nearby", m_iMaxNearbyEntities);
        m_bCaptureNearbyEntities = moduleConfig.GetBoolValue("capture_nearby", m_bCaptureNearbyEntities);
        m_bCaptureTerrainInfo = moduleConfig.GetBoolValue("capture_terrain", m_bCaptureTerrainInfo);

        // Cache weather manager
        CacheWeatherManager();

        Print("[SceneModule] Initialized (radius=" + m_fNearbySearchRadius.ToString() + "m)", LogLevel.NORMAL);
        return SCR_CaptureResult.Success();
    }

    //------------------------------------------------------------------------
    protected void CacheWeatherManager()
    {
        if (m_bWeatherManagerCached)
            return;

        ChimeraWorld world = ChimeraWorld.CastFrom(GetGame().GetWorld());
        if (world)
        {
            m_WeatherManager = world.GetTimeAndWeatherManager();
        }

        m_bWeatherManagerCached = true;
    }

    //------------------------------------------------------------------------
    override string GetCSVHeader()
    {
        return SCR_SceneContextRecord.GetCSVHeader();
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

        // Capture scene context (shared across targets)
        CacheWeatherManager();

        // Capture for each target
        array<IEntity> targets = context.GetTargets();
        for (int t = 0; t < targets.Count(); t++)
        {
            IEntity target = targets[t];
            if (!target)
                continue;

            // Capture scene context
            SCR_SceneContextRecord record = CaptureSceneContext(target, frameId, timestampMs, t);
            if (record)
            {
                buffer.Write(record, timestampMs);
                bytesWritten += 100;
                capturedCount++;
            }

            // Capture nearby entities if enabled
            if (m_bCaptureNearbyEntities)
            {
                int nearbyBytes = CaptureNearbyEntities(target, frameId, timestampMs, t, buffer);
                bytesWritten += nearbyBytes;
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
    protected SCR_SceneContextRecord CaptureSceneContext(IEntity target, int frameId, float timestampMs, int targetIndex)
    {
        SCR_SceneContextRecord record = new SCR_SceneContextRecord(frameId, timestampMs, targetIndex);

        // Get target position
        vector transform[4];
        target.GetWorldTransform(transform);
        vector position = transform[3];

        // Time of day
        if (m_WeatherManager)
        {
            float hours, minutes, seconds;
            m_WeatherManager.GetHoursMinutesSeconds(hours, minutes, seconds);
            record.m_fTimeOfDay = hours + minutes / 60.0 + seconds / 3600.0;

            // Estimate sun angle (simplified)
            float solarTime = record.m_fTimeOfDay;
            if (solarTime < 6 || solarTime > 18)
            {
                record.m_fSunAngle = 0;  // Night
            }
            else
            {
                // Approximate sun angle (peaks at noon)
                record.m_fSunAngle = 90.0 * (1.0 - Math.AbsFloat(12.0 - solarTime) / 6.0);
            }

            // Weather state - try to get current weather
            // Note: Actual API may vary, this is a simplified approach
            WeatherState currentWeather = m_WeatherManager.GetCurrentWeatherState();
            if (currentWeather)
            {
                // Map weather to enum (simplified)
                string weatherName = currentWeather.GetStateName();
                record.m_iWeatherState = ClassifyWeather(weatherName);

                // Get weather parameters
                record.m_fFogDensity = currentWeather.GetFogDensity();
                record.m_fRainIntensity = currentWeather.GetRainIntensity();
            }
        }

        // Nearby entity counts
        if (m_bCaptureNearbyEntities)
        {
            CountNearbyEntities(position, record);
        }

        // Terrain info
        if (m_bCaptureTerrainInfo)
        {
            CaptureTerrainInfo(position, record);
        }

        return record;
    }

    //------------------------------------------------------------------------
    protected int ClassifyWeather(string weatherName)
    {
        weatherName.ToLower();

        if (weatherName.Contains("clear") || weatherName.Contains("sunny"))
            return SCR_WeatherState.WEATHER_CLEAR;
        if (weatherName.Contains("cloud"))
            return SCR_WeatherState.WEATHER_CLOUDY;
        if (weatherName.Contains("overcast"))
            return SCR_WeatherState.WEATHER_OVERCAST;
        if (weatherName.Contains("rain"))
            return SCR_WeatherState.WEATHER_RAINY;
        if (weatherName.Contains("storm") || weatherName.Contains("thunder"))
            return SCR_WeatherState.WEATHER_STORMY;
        if (weatherName.Contains("fog"))
            return SCR_WeatherState.WEATHER_FOGGY;
        if (weatherName.Contains("snow"))
            return SCR_WeatherState.WEATHER_SNOWY;

        return SCR_WeatherState.WEATHER_UNKNOWN;
    }

    //------------------------------------------------------------------------
    protected void CountNearbyEntities(vector position, SCR_SceneContextRecord record)
    {
        record.m_iNearbyVehicles = 0;
        record.m_iNearbyCharacters = 0;
        record.m_iNearbyObjects = 0;
        record.m_fNearestObstacle = m_fNearbySearchRadius;

        // Get nearby entities
        ref array<IEntity> nearbyEntities = new array<IEntity>();
        GetGame().GetWorld().QueryEntitiesBySphere(position, m_fNearbySearchRadius, null, nearbyEntities);

        for (int i = 0; i < nearbyEntities.Count() && i < m_iMaxNearbyEntities; i++)
        {
            IEntity entity = nearbyEntities[i];
            if (!entity)
                continue;

            float distance = vector.Distance(position, entity.GetOrigin());

            // Classify entity
            VehicleWheeledSimulation vehicleSim = VehicleWheeledSimulation.Cast(entity.FindComponent(VehicleWheeledSimulation));
            if (vehicleSim)
            {
                record.m_iNearbyVehicles++;
                if (distance < record.m_fNearestObstacle)
                    record.m_fNearestObstacle = distance;
                continue;
            }

            // Check for characters (simplified)
            string className = entity.ClassName();
            if (className.Contains("Character") || className.Contains("Soldier") || className.Contains("Man"))
            {
                record.m_iNearbyCharacters++;
                continue;
            }

            // Other objects
            record.m_iNearbyObjects++;
            if (distance < record.m_fNearestObstacle && distance > 0.5)  // Ignore self
                record.m_fNearestObstacle = distance;
        }

        if (record.m_fNearestObstacle >= m_fNearbySearchRadius)
            record.m_fNearestObstacle = -1;  // No obstacle found
    }

    //------------------------------------------------------------------------
    protected int CaptureNearbyEntities(IEntity target, int frameId, float timestampMs, int targetIndex, SCR_CaptureBuffer buffer)
    {
        vector transform[4];
        target.GetWorldTransform(transform);
        vector position = transform[3];
        vector forward = transform[2];

        int bytesWritten = 0;

        // Query nearby entities
        ref array<IEntity> nearbyEntities = new array<IEntity>();
        GetGame().GetWorld().QueryEntitiesBySphere(position, m_fNearbySearchRadius, null, nearbyEntities);

        int entityIndex = 0;
        for (int i = 0; i < nearbyEntities.Count() && entityIndex < m_iMaxNearbyEntities; i++)
        {
            IEntity entity = nearbyEntities[i];
            if (!entity || entity == target)
                continue;

            vector entityPos = entity.GetOrigin();
            vector relative = entityPos - position;
            float distance = relative.Length();

            if (distance < 0.5)  // Skip very close/self
                continue;

            SCR_NearbyEntityRecord record = new SCR_NearbyEntityRecord(frameId, timestampMs, targetIndex, entityIndex);

            record.m_fRelativeX = relative[0];
            record.m_fRelativeY = relative[1];
            record.m_fRelativeZ = relative[2];
            record.m_fDistance = distance;

            // Calculate bearing (angle from forward direction)
            vector relativeNorm = relative;
            relativeNorm.Normalize();
            float dot = vector.Dot(forward, relativeNorm);
            record.m_fBearing = Math.Acos(dot) * Math.RAD2DEG;

            // Determine side (left/right)
            vector cross = forward * relativeNorm;
            if (cross[1] < 0)
                record.m_fBearing = -record.m_fBearing;

            // Classify entity type
            VehicleWheeledSimulation vehicleSim = VehicleWheeledSimulation.Cast(entity.FindComponent(VehicleWheeledSimulation));
            if (vehicleSim)
            {
                record.m_sEntityType = "vehicle";
                record.m_fVelocity = vehicleSim.GetSpeedKmh();
            }
            else
            {
                record.m_sEntityType = "object";
                record.m_fVelocity = 0;
            }

            buffer.Write(record, timestampMs);
            bytesWritten += 60;
            entityIndex++;
        }

        return bytesWritten;
    }

    //------------------------------------------------------------------------
    protected void CaptureTerrainInfo(vector position, SCR_SceneContextRecord record)
    {
        record.m_sTerrainType = "unknown";
        record.m_fTerrainSlope = 0;

        // Sample terrain at multiple points to estimate slope
        float sampleDist = 2.0;
        float centerHeight = GetTerrainHeight(position);

        float frontHeight = GetTerrainHeight(position + "0 0 " + sampleDist.ToString());
        float backHeight = GetTerrainHeight(position + "0 0 -" + sampleDist.ToString());
        float leftHeight = GetTerrainHeight(position + "-" + sampleDist.ToString() + " 0 0");
        float rightHeight = GetTerrainHeight(position + sampleDist.ToString() + " 0 0");

        // Calculate approximate slope
        float slopeFB = Math.Atan2(frontHeight - backHeight, sampleDist * 2.0) * Math.RAD2DEG;
        float slopeLR = Math.Atan2(rightHeight - leftHeight, sampleDist * 2.0) * Math.RAD2DEG;

        record.m_fTerrainSlope = Math.Sqrt(slopeFB * slopeFB + slopeLR * slopeLR);

        // Classify terrain based on slope
        if (record.m_fTerrainSlope < 5)
            record.m_sTerrainType = "flat";
        else if (record.m_fTerrainSlope < 15)
            record.m_sTerrainType = "mild_slope";
        else if (record.m_fTerrainSlope < 30)
            record.m_sTerrainType = "steep";
        else
            record.m_sTerrainType = "cliff";
    }

    //------------------------------------------------------------------------
    protected float GetTerrainHeight(vector position)
    {
        // Raycast down to get terrain height
        TraceParam trace = new TraceParam();
        trace.Start = Vector(position[0], position[1] + 100, position[2]);
        trace.End = Vector(position[0], position[1] - 100, position[2]);
        trace.Flags = TraceFlags.WORLD;
        trace.LayerMask = TRACE_LAYER_CAMERA;

        float fraction = GetGame().GetWorld().TraceMove(trace, null);
        if (fraction < 1.0)
        {
            return trace.Start[1] - fraction * 200.0;
        }

        return position[1];
    }

    //------------------------------------------------------------------------
    override SCR_CaptureResult Finalize()
    {
        Print("[SceneModule] Finalized. Total captures: " + m_iTotalCaptureCount.ToString(), LogLevel.NORMAL);
        return super.Finalize();
    }
}
