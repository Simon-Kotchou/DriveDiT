// ============================================================================
// SCR_AIDrivingSimulator - AI Driving Simulator with ML Data Collection
// ============================================================================
//
// Complete AI driving simulation system that spawns AI groups, manages
// waypoints, tracks vehicle telemetry, and captures ML training data.
//
// Designed to run as a game mode component on SCR_BaseGameMode or GameMode_Plain.
//
// USAGE:
//   1. Add this component to your GameMode entity
//   2. Configure spawn points (AI_Spawn_1, AI_Spawn_2, etc.)
//   3. Configure drive targets (AI_Drive_Target_1 through AI_Drive_Target_20)
//   4. Configure vehicles (AI_Car_1, AI_Car_2, etc.)
//   5. Run the scenario
//
// OUTPUT:
//   $profile:DrivingData/session_XXXX/telemetry.csv
//   $profile:DrivingData/session_XXXX/session_info.txt
//   $profile:DrivingData/session_XXXX/summary.txt
//
// ============================================================================

[ComponentEditorProps(category: "GameScripted/GameMode", description: "AI Driving Simulator with ML Data Collection")]
class SCR_AIDrivingSimulatorClass: SCR_BaseGameModeComponentClass
{
}

class SCR_AIDrivingSimulator: SCR_BaseGameModeComponent
{
    // === CONFIGURATION ===
    [Attribute("1", UIWidgets.Slider, "Maximum number of AI groups to spawn", "1 10 1")]
    protected int m_iMaxAIGroups;

    [Attribute("30.0", UIWidgets.Slider, "Waypoint completion radius in meters", "10 100 5")]
    protected float m_fWaypointRadius;

    [Attribute("3000", UIWidgets.Slider, "Delay between waypoint completions (ms)", "1000 10000 500")]
    protected int m_iWaypointDelay;

    [Attribute("1", UIWidgets.CheckBox, "Enable detailed telemetry logging")]
    protected bool m_bDetailedTelemetry;

    [Attribute("1", UIWidgets.CheckBox, "Monitor and log AI inputs")]
    protected bool m_bMonitorAIInputs;

    [Attribute("Clear", UIWidgets.EditBox, "Initial weather state")]
    protected string m_sInitialWeather;

    [Attribute("12.0", UIWidgets.Slider, "Initial time of day (24h)", "0 24 0.5")]
    protected float m_fInitialTime;

    // === DATA CAPTURE CONFIG ===
    [Attribute("1", UIWidgets.CheckBox, "Enable ML data capture")]
    protected bool m_bEnableDataCapture;

    [Attribute("200", UIWidgets.Slider, "Capture interval (ms) - 200ms = 5Hz", "50 1000 50")]
    protected int m_iCaptureIntervalMs;

    // === INTERNAL STATE ===
    protected ref array<SCR_AIGroup> m_aActiveGroups;
    protected ref array<int> m_aGroupIDs;
    protected ref array<string> m_aVehicleNames;
    protected ref array<ref array<AIWaypoint>> m_aGroupWaypoints;
    protected ref array<int> m_aCurrentWaypointIndices;
    protected ref array<int> m_aWaypointsCompleted;
    protected ref array<float> m_aLastWaypointTimes;

    // Vehicle tracking
    protected ref array<IEntity> m_aVehicles;
    protected ref array<VehicleWheeledSimulation> m_aVehicleSimulations;
    protected ref array<float> m_aVehicleDistances;
    protected ref array<float> m_aMaxSpeeds;
    protected ref array<float> m_aMaxRPMs;
    protected ref array<vector> m_aLastPositions;
    protected ref array<float> m_aLastSpeeds;
    protected ref array<float> m_aAccelerations;

    // Stuck detection
    protected ref array<vector> m_aStuckCheckPositions;
    protected ref array<float> m_aStuckTimers;
    static const float STUCK_TIME_THRESHOLD = 30.0;
    static const float STUCK_DISTANCE_THRESHOLD = 5.0;

    // Waypoint classification
    protected ref array<int> m_aWaypointTypes;
    protected ref array<string> m_aWaypointDescriptions;
    protected ref array<float> m_aSpeedLimits;
    protected ref array<AIWaypoint> m_aGlobalWaypoints;

    // Waypoint type constants
    static const int WAYPOINT_TYPE_CITY = 0;
    static const int WAYPOINT_TYPE_HIGHWAY = 1;
    static const int WAYPOINT_TYPE_RURAL = 2;
    static const int WAYPOINT_TYPE_OFFROAD = 3;

    // Environment
    protected TimeAndWeatherManagerEntity m_WeatherManager;
    protected float m_fSimulationStartTime;
    protected int m_iTotalWaypointsCompleted;

    // Data capture
    protected SCR_MLDataCollector m_DataCollector;
    protected bool m_bDataCaptureInitialized;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        // Initialize all arrays
        m_aActiveGroups = new array<SCR_AIGroup>();
        m_aGroupIDs = new array<int>();
        m_aVehicleNames = new array<string>();
        m_aGroupWaypoints = new array<ref array<AIWaypoint>>();
        m_aCurrentWaypointIndices = new array<int>();
        m_aWaypointsCompleted = new array<int>();
        m_aLastWaypointTimes = new array<float>();

        m_aVehicles = new array<IEntity>();
        m_aVehicleSimulations = new array<VehicleWheeledSimulation>();
        m_aVehicleDistances = new array<float>();
        m_aMaxSpeeds = new array<float>();
        m_aMaxRPMs = new array<float>();
        m_aLastPositions = new array<vector>();
        m_aLastSpeeds = new array<float>();
        m_aAccelerations = new array<float>();

        m_aStuckCheckPositions = new array<vector>();
        m_aStuckTimers = new array<float>();

        m_aWaypointTypes = new array<int>();
        m_aWaypointDescriptions = new array<string>();
        m_aSpeedLimits = new array<float>();
        m_aGlobalWaypoints = new array<AIWaypoint>();

        m_iTotalWaypointsCompleted = 0;
        m_bDataCaptureInitialized = false;

        Print("[AIDrivingSim] Component initialized", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    override void OnGameModeStart()
    {
        super.OnGameModeStart();

        m_fSimulationStartTime = GetGame().GetWorld().GetWorldTime();

        Print("[AIDrivingSim] Starting AI driving simulation", LogLevel.NORMAL);

        // Initialize environment
        InitializeEnvironment();

        // Classify waypoints
        InitializeWaypointClassifications();

        // Initialize data collector if enabled
        if (m_bEnableDataCapture)
        {
            InitializeDataCollector();
        }

        // Spawn AI groups with staggered timing
        for (int i = 1; i <= m_iMaxAIGroups; i++)
        {
            GetGame().GetCallqueue().CallLater(SpawnAIGroup, i * 2000, false, i);
        }

        // Start telemetry monitoring
        if (m_bDetailedTelemetry)
        {
            GetGame().GetCallqueue().CallLater(MonitorVehicleTelemetry, m_iCaptureIntervalMs, true);
        }

        // Statistics reporting every 30 seconds
        GetGame().GetCallqueue().CallLater(ReportDrivingStatistics, 30000, true);

        // Initialize data capture after vehicles spawn
        if (m_bEnableDataCapture)
        {
            GetGame().GetCallqueue().CallLater(StartDataCapture, (m_iMaxAIGroups + 1) * 2000 + 1000, false);
        }
    }

    //------------------------------------------------------------------------
    protected void InitializeEnvironment()
    {
        ChimeraWorld world = ChimeraWorld.CastFrom(GetGame().GetWorld());
        if (!world)
            return;

        m_WeatherManager = world.GetTimeAndWeatherManager();
        if (!m_WeatherManager)
            return;

        // Set time of day
        m_WeatherManager.SetHoursMinutesSeconds(m_fInitialTime, 0, 0, true);

        // Set weather
        if (!m_sInitialWeather.IsEmpty())
        {
            m_WeatherManager.ForceWeatherTo(true, m_sInitialWeather, 0.1, 999999);
        }

        Print("[AIDrivingSim] Environment: Time=" + m_fInitialTime.ToString(4, 1) + ", Weather=" + m_sInitialWeather, LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void InitializeWaypointClassifications()
    {
        m_aWaypointTypes.Clear();
        m_aWaypointDescriptions.Clear();
        m_aSpeedLimits.Clear();

        // Classify 20 waypoints into road types
        for (int i = 1; i <= 20; i++)
        {
            int waypointType;
            string description;
            float speedLimit;

            if (i <= 6)
            {
                waypointType = WAYPOINT_TYPE_CITY;
                description = "City Street";
                speedLimit = 50.0;
            }
            else if (i <= 12)
            {
                waypointType = WAYPOINT_TYPE_HIGHWAY;
                description = "Highway";
                speedLimit = 120.0;
            }
            else if (i <= 17)
            {
                waypointType = WAYPOINT_TYPE_RURAL;
                description = "Rural Road";
                speedLimit = 80.0;
            }
            else
            {
                waypointType = WAYPOINT_TYPE_OFFROAD;
                description = "Off-Road";
                speedLimit = 40.0;
            }

            m_aWaypointTypes.Insert(waypointType);
            m_aWaypointDescriptions.Insert(description);
            m_aSpeedLimits.Insert(speedLimit);
        }

        Print("[AIDrivingSim] Classified 20 waypoints into 4 categories", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void InitializeDataCollector()
    {
        // Try to find existing data collector component
        IEntity owner = GetOwner();
        if (!owner)
            return;

        m_DataCollector = SCR_MLDataCollector.Cast(owner.FindComponent(SCR_MLDataCollector));

        if (!m_DataCollector)
        {
            Print("[AIDrivingSim] WARNING: No SCR_MLDataCollector component found. Add one to enable data capture.", LogLevel.WARNING);
            m_bEnableDataCapture = false;
        }
    }

    //------------------------------------------------------------------------
    protected void StartDataCapture()
    {
        if (!m_DataCollector || !m_bEnableDataCapture)
            return;

        // Initialize session
        if (!m_DataCollector.InitializeSession())
        {
            Print("[AIDrivingSim] ERROR: Failed to initialize data capture session", LogLevel.ERROR);
            return;
        }

        // Set waypoint classifications
        m_DataCollector.SetWaypointClassifications(m_aWaypointTypes, m_aSpeedLimits, m_aWaypointDescriptions);

        // Register all tracked vehicles
        for (int i = 0; i < m_aVehicles.Count(); i++)
        {
            IEntity vehicle = m_aVehicles[i];
            if (vehicle)
            {
                int waypointIdx = -1;
                if (i < m_aCurrentWaypointIndices.Count())
                    waypointIdx = m_aCurrentWaypointIndices[i];

                m_DataCollector.RegisterVehicle(vehicle, waypointIdx);
            }
        }

        // Write session info
        m_DataCollector.WriteSessionInfo("GM_Arland", m_sInitialWeather, m_fInitialTime);

        m_bDataCaptureInitialized = true;
        Print("[AIDrivingSim] Data capture started", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void SpawnAIGroup(int groupIndex)
    {
        string spawnGroup = "{35681BE27C302FF5}Prefabs/Groups/BLUFOR/GreenBerets/Group_US_GreenBeret_SentryTeam.et";

        // Find spawn position
        string spawnPointName = "AI_Spawn_" + groupIndex.ToString();
        IEntity spawnPosition = GetGame().GetWorld().FindEntityByName(spawnPointName);

        if (!spawnPosition)
            spawnPosition = GetGame().GetWorld().FindEntityByName("AI_Spawn_1");

        if (!spawnPosition)
        {
            Print("[AIDrivingSim] ERROR: No spawn position for group " + groupIndex.ToString(), LogLevel.ERROR);
            return;
        }

        // Load and spawn group
        Resource resource = Resource.Load(spawnGroup);
        if (!resource || !resource.IsValid())
        {
            Print("[AIDrivingSim] ERROR: Invalid group resource", LogLevel.ERROR);
            return;
        }

        EntitySpawnParams params = new EntitySpawnParams();
        params.TransformMode = ETransformMode.WORLD;
        params.Transform[3] = spawnPosition.GetOrigin();

        SCR_AIGroup group = SCR_AIGroup.Cast(GetGame().SpawnEntityPrefab(resource, null, params));
        if (!group)
        {
            Print("[AIDrivingSim] ERROR: Failed to spawn AI group", LogLevel.ERROR);
            return;
        }

        // Track the group
        m_aActiveGroups.Insert(group);
        m_aGroupIDs.Insert(groupIndex);

        string vehicleName = "AI_Car_" + groupIndex.ToString();
        m_aVehicleNames.Insert(vehicleName);
        m_aCurrentWaypointIndices.Insert(-1);
        m_aWaypointsCompleted.Insert(0);
        m_aLastWaypointTimes.Insert(GetGame().GetWorld().GetWorldTime());

        // Find and assign vehicle
        IEntity vehicle = GetGame().GetWorld().FindEntityByName(vehicleName);
        if (!vehicle)
        {
            Print("[AIDrivingSim] WARNING: Vehicle " + vehicleName + " not found, using AI_Car_1", LogLevel.WARNING);
            vehicle = GetGame().GetWorld().FindEntityByName("AI_Car_1");
            vehicleName = "AI_Car_1";
        }

        if (vehicle)
        {
            ref array<string> vehicleArray = new array<string>();
            vehicleArray.Insert(vehicleName);
            group.AddVehiclesStatic(vehicleArray);

            if (m_bDetailedTelemetry)
                InitializeVehicleTelemetry(vehicle, groupIndex);
        }

        // Create waypoints for this group
        CreateWaypointsForGroup(groupIndex);

        // Hook waypoint completion
        group.GetOnWaypointCompleted().Insert(OnWaypointCompleted);

        // Start with first waypoint
        AddNextWaypointForGroup(groupIndex);

        Print("[AIDrivingSim] Group " + groupIndex.ToString() + " spawned with vehicle " + vehicleName, LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void InitializeVehicleTelemetry(IEntity vehicle, int groupID)
    {
        VehicleWheeledSimulation vehicleSim = VehicleWheeledSimulation.Cast(vehicle.FindComponent(VehicleWheeledSimulation));
        if (!vehicleSim)
        {
            Print("[AIDrivingSim] ERROR: No VehicleWheeledSimulation on vehicle", LogLevel.ERROR);
            return;
        }

        m_aVehicles.Insert(vehicle);
        m_aVehicleSimulations.Insert(vehicleSim);
        m_aVehicleDistances.Insert(0);
        m_aMaxSpeeds.Insert(0);
        m_aMaxRPMs.Insert(0);
        m_aLastSpeeds.Insert(0);
        m_aAccelerations.Insert(0);

        vector transform[4];
        vehicle.GetWorldTransform(transform);
        m_aLastPositions.Insert(transform[3]);
        m_aStuckCheckPositions.Insert(transform[3]);
        m_aStuckTimers.Insert(0);

        Print("[AIDrivingSim] Telemetry initialized for Group " + groupID.ToString(), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void CreateWaypointsForGroup(int groupIndex)
    {
        ref array<AIWaypoint> groupWaypoints = new array<AIWaypoint>();

        string waypointPrefab = "{06E1B6EBD480C6E0}Prefabs/AI/Waypoints/AIWaypoint_ForcedMove.et";
        Resource resource = Resource.Load(waypointPrefab);
        if (!resource || !resource.IsValid())
        {
            Print("[AIDrivingSim] ERROR: Invalid waypoint prefab", LogLevel.ERROR);
            return;
        }

        for (int i = 1; i <= 20; i++)
        {
            string placeholderName = "AI_Drive_Target_" + i.ToString();
            IEntity placeholderPosition = GetGame().GetWorld().FindEntityByName(placeholderName);

            if (placeholderPosition)
            {
                EntitySpawnParams params = new EntitySpawnParams();
                params.TransformMode = ETransformMode.WORLD;
                params.Transform[3] = placeholderPosition.GetOrigin();

                AIWaypoint waypoint = AIWaypoint.Cast(GetGame().SpawnEntityPrefab(resource, null, params));
                if (waypoint)
                {
                    waypoint.SetCompletionRadius(m_fWaypointRadius);
                    waypoint.SetName("DrivingWP_" + groupIndex.ToString() + "_" + i.ToString());
                    groupWaypoints.Insert(waypoint);
                    m_aGlobalWaypoints.Insert(waypoint);
                }
            }
        }

        m_aGroupWaypoints.Insert(groupWaypoints);
        Print("[AIDrivingSim] Created " + groupWaypoints.Count().ToString() + " waypoints for group " + groupIndex.ToString(), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void AddNextWaypointForGroup(int groupIndex)
    {
        if (groupIndex < 1 || groupIndex > m_aActiveGroups.Count())
            return;

        int arrayIndex = groupIndex - 1;
        SCR_AIGroup group = m_aActiveGroups[arrayIndex];
        ref array<AIWaypoint> groupWaypoints = m_aGroupWaypoints[arrayIndex];

        if (!group || !groupWaypoints || groupWaypoints.IsEmpty())
            return;

        // Pick random waypoint
        int randomIndex = Math.RandomIntInclusive(0, groupWaypoints.Count() - 1);
        AIWaypoint nextWaypoint = groupWaypoints[randomIndex];

        if (nextWaypoint)
        {
            group.AddWaypoint(nextWaypoint);
            m_aCurrentWaypointIndices[arrayIndex] = randomIndex;

            // Update data collector
            if (m_DataCollector && arrayIndex < m_aVehicles.Count())
            {
                m_DataCollector.SetVehicleWaypoint(arrayIndex, randomIndex);
            }

            string description = "Unknown";
            float speedLimit = 0;
            if (randomIndex < m_aWaypointDescriptions.Count())
                description = m_aWaypointDescriptions[randomIndex];
            if (randomIndex < m_aSpeedLimits.Count())
                speedLimit = m_aSpeedLimits[randomIndex];

            Print("[AIDrivingSim] Group " + groupIndex.ToString() + " heading to " + description + " (limit: " + speedLimit.ToString(5, 1) + " km/h)", LogLevel.NORMAL);
        }
    }

    //------------------------------------------------------------------------
    protected void OnWaypointCompleted(AIWaypoint completedWaypoint)
    {
        m_iTotalWaypointsCompleted++;

        int groupIndex = FindGroupIndexByWaypoint(completedWaypoint);
        if (groupIndex < 0)
            return;

        int waypointIndex = m_aCurrentWaypointIndices[groupIndex];
        string waypointType = "Unknown";
        if (waypointIndex >= 0 && waypointIndex < m_aWaypointDescriptions.Count())
            waypointType = m_aWaypointDescriptions[waypointIndex];

        Print("[AIDrivingSim] Group " + m_aGroupIDs[groupIndex].ToString() + " completed " + waypointType + " waypoint", LogLevel.NORMAL);

        m_aWaypointsCompleted[groupIndex] = m_aWaypointsCompleted[groupIndex] + 1;
        m_aLastWaypointTimes[groupIndex] = GetGame().GetWorld().GetWorldTime();

        SCR_AIGroup group = m_aActiveGroups[groupIndex];
        group.RemoveWaypoint(completedWaypoint);

        int actualGroupID = m_aGroupIDs[groupIndex];
        GetGame().GetCallqueue().CallLater(AddNextWaypointForGroup, m_iWaypointDelay, false, actualGroupID);
    }

    //------------------------------------------------------------------------
    protected int FindGroupIndexByWaypoint(AIWaypoint waypoint)
    {
        for (int i = 0; i < m_aGroupWaypoints.Count(); i++)
        {
            ref array<AIWaypoint> groupWaypoints = m_aGroupWaypoints[i];
            if (groupWaypoints && groupWaypoints.Find(waypoint) >= 0)
                return i;
        }
        return -1;
    }

    //------------------------------------------------------------------------
    protected void MonitorVehicleTelemetry()
    {
        for (int i = 0; i < m_aVehicleSimulations.Count(); i++)
        {
            VehicleWheeledSimulation sim = m_aVehicleSimulations[i];
            IEntity vehicle = m_aVehicles[i];

            if (!sim || !vehicle)
                continue;

            UpdateVehicleTelemetry(i, sim, vehicle);
        }

        // Trigger data capture
        if (m_DataCollector && m_bDataCaptureInitialized)
        {
            m_DataCollector.CaptureFrame();
        }
    }

    //------------------------------------------------------------------------
    protected void UpdateVehicleTelemetry(int index, VehicleWheeledSimulation sim, IEntity vehicle)
    {
        float speed = sim.GetSpeedKmh();

        vector transform[4];
        vehicle.GetWorldTransform(transform);
        vector currentPosition = transform[3];

        // Update distance
        vector lastPosition = m_aLastPositions[index];
        float distance = vector.Distance(currentPosition, lastPosition);
        m_aVehicleDistances[index] = m_aVehicleDistances[index] + distance;
        m_aLastPositions[index] = currentPosition;

        // Update acceleration
        float lastSpeed = m_aLastSpeeds[index];
        float intervalSec = m_iCaptureIntervalMs / 1000.0;
        float acceleration = (speed - lastSpeed) / intervalSec;
        m_aAccelerations[index] = acceleration;
        m_aLastSpeeds[index] = speed;

        // Update max stats
        if (speed > m_aMaxSpeeds[index])
            m_aMaxSpeeds[index] = speed;

        float engineRPM = sim.EngineGetRPM();
        if (engineRPM > m_aMaxRPMs[index])
            m_aMaxRPMs[index] = engineRPM;

        // Check stuck
        CheckStuckVehicle(index, currentPosition);
    }

    //------------------------------------------------------------------------
    protected void CheckStuckVehicle(int index, vector currentPosition)
    {
        vector stuckCheckPosition = m_aStuckCheckPositions[index];
        float distanceMoved = vector.Distance(currentPosition, stuckCheckPosition);

        if (distanceMoved < STUCK_DISTANCE_THRESHOLD)
        {
            float intervalSec = m_iCaptureIntervalMs / 1000.0;
            m_aStuckTimers[index] = m_aStuckTimers[index] + intervalSec;

            if (m_aStuckTimers[index] >= STUCK_TIME_THRESHOLD)
            {
                ResetStuckVehicle(index);
            }
        }
        else
        {
            m_aStuckTimers[index] = 0;
            m_aStuckCheckPositions[index] = currentPosition;
        }
    }

    //------------------------------------------------------------------------
    protected void ResetStuckVehicle(int index)
    {
        IEntity vehicle = m_aVehicles[index];
        if (!vehicle)
            return;

        int groupID = index + 1;
        Print("[AIDrivingSim] Group " + groupID.ToString() + " stuck for " + STUCK_TIME_THRESHOLD.ToString(4, 1) + "s - resetting", LogLevel.WARNING);

        string spawnPointName = "AI_Spawn_" + groupID.ToString();
        IEntity spawnPosition = GetGame().GetWorld().FindEntityByName(spawnPointName);

        if (!spawnPosition)
            spawnPosition = GetGame().GetWorld().FindEntityByName("AI_Spawn_1");

        if (spawnPosition)
        {
            vector spawnTransform[4];
            spawnPosition.GetWorldTransform(spawnTransform);
            spawnTransform[3][1] = spawnTransform[3][1] + 2.0;

            vehicle.SetWorldTransform(spawnTransform);

            m_aStuckTimers[index] = 0;
            m_aStuckCheckPositions[index] = spawnTransform[3];
            m_aLastPositions[index] = spawnTransform[3];

            Print("[AIDrivingSim] Group " + groupID.ToString() + " reset to spawn position", LogLevel.NORMAL);
        }
    }

    //------------------------------------------------------------------------
    protected void ReportDrivingStatistics()
    {
        float simulationTime = (GetGame().GetWorld().GetWorldTime() - m_fSimulationStartTime) / 1000.0;

        Print("=== DRIVING SIMULATION REPORT ===", LogLevel.NORMAL);
        Print("Simulation time: " + (simulationTime / 60.0).ToString(6, 1) + " minutes", LogLevel.NORMAL);
        Print("Active groups: " + m_aActiveGroups.Count().ToString(), LogLevel.NORMAL);
        Print("Total waypoints: " + m_iTotalWaypointsCompleted.ToString(), LogLevel.NORMAL);

        if (m_DataCollector && m_bDataCaptureInitialized)
        {
            Print("Data capture frames: " + m_DataCollector.GetFrameCount().ToString(), LogLevel.NORMAL);
            Print("Session path: " + m_DataCollector.GetSessionPath(), LogLevel.NORMAL);
        }

        for (int i = 0; i < m_aGroupIDs.Count(); i++)
        {
            int groupID = m_aGroupIDs[i];
            int waypointsCompleted = m_aWaypointsCompleted[i];

            float distance = 0;
            float maxSpeed = 0;
            float maxRPM = 0;

            if (i < m_aVehicleDistances.Count())
            {
                distance = m_aVehicleDistances[i];
                maxSpeed = m_aMaxSpeeds[i];
                maxRPM = m_aMaxRPMs[i];
            }

            Print("Group " + groupID.ToString() + ": WP=" + waypointsCompleted.ToString() + " Dist=" + distance.ToString(8, 1) + "m MaxSpd=" + maxSpeed.ToString(5, 1) + "km/h", LogLevel.NORMAL);
        }

        Print("==================================", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    override void OnGameModeEnd(SCR_GameModeEndData data)
    {
        // Finalize data capture
        if (m_DataCollector && m_bDataCaptureInitialized)
        {
            m_DataCollector.FinalizeSession();
        }

        super.OnGameModeEnd(data);
    }
}
