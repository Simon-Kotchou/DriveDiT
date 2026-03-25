// ============================================================================
// SCR_MLDataCollector - ML Training Data Collection Component
// ============================================================================
//
// Captures synchronized telemetry data for world model training.
// Outputs CSV format compatible with comma.ai/Open-Oasis training pipelines.
//
// OUTPUT:
//   $profile:DrivingData/session_XXXX/telemetry.csv
//   $profile:DrivingData/session_XXXX/session_info.txt
//   $profile:DrivingData/session_XXXX/anchors.csv (when anchor selection enabled)
//   $profile:DrivingData/session_XXXX/frames/frame_XXXXXX.bmp (when screenshot capture enabled)
//
// INTEGRATIONS:
//   - SCR_AnchorFrameSelector: Anchor frame selection for Self-Forcing++ training
//   - SCR_MLScreenshotCapture: Synchronized frame capture for visual data
//
// ============================================================================

[ComponentEditorProps(category: "GameScripted/DataCapture", description: "ML Training Data Collection - Captures vehicle telemetry for world model training")]
class SCR_MLDataCollectorClass: ScriptComponentClass
{
}

class SCR_MLDataCollector: ScriptComponent
{
    // === CONFIGURATION ===
    [Attribute("1", UIWidgets.CheckBox, "Enable data capture")]
    protected bool m_bEnableDataCapture;

    [Attribute("200", UIWidgets.Slider, "Capture interval in milliseconds (200ms = 5Hz)", "50 1000 50")]
    protected int m_iCaptureIntervalMs;

    [Attribute("0", UIWidgets.CheckBox, "Capture screenshots (requires external tool sync)")]
    protected bool m_bCaptureScreenshots;

    [Attribute("1", UIWidgets.CheckBox, "Log detailed telemetry to console")]
    protected bool m_bVerboseLogging;

    [Attribute("500", UIWidgets.Slider, "Progress log interval (frames)", "100 2000 100")]
    protected int m_iProgressLogInterval;

    // === ANCHOR FRAME CONFIGURATION ===
    [Attribute("1", UIWidgets.CheckBox, "Enable anchor frame selection for Self-Forcing++ training")]
    protected bool m_bEnableAnchorSelection;

    // === SESSION STATE ===
    protected string m_sSessionPath;
    protected string m_sTelemetryPath;
    protected int m_iFrameCounter;
    protected bool m_bSessionInitialized;
    protected float m_fSessionStartTime;
    protected float m_fLastCaptureTime;

    // === VEHICLE TRACKING ===
    protected ref array<IEntity> m_aTrackedVehicles;
    protected ref array<VehicleWheeledSimulation> m_aVehicleSimulations;
    protected ref array<vector> m_aLastPositions;
    protected ref array<float> m_aLastSpeeds;
    protected ref array<float> m_aDistances;
    protected ref array<float> m_aAccelerations;

    // === WAYPOINT CONTEXT ===
    protected ref array<int> m_aCurrentWaypointIndices;
    protected ref array<int> m_aWaypointTypes;
    protected ref array<float> m_aSpeedLimits;
    protected ref array<string> m_aWaypointDescriptions;

    // === ANCHOR FRAME SELECTION ===
    protected SCR_AnchorFrameSelector m_AnchorSelector;
    protected bool m_bAnchorSelectorInitialized;
    protected int m_iAnchorFrameCount;

    // === SCREENSHOT CAPTURE ===
    protected SCR_MLScreenshotCapture m_ScreenshotCapture;
    protected bool m_bScreenshotCaptureInitialized;

    // Waypoint type constants
    static const int WAYPOINT_TYPE_CITY = 0;
    static const int WAYPOINT_TYPE_HIGHWAY = 1;
    static const int WAYPOINT_TYPE_RURAL = 2;
    static const int WAYPOINT_TYPE_OFFROAD = 3;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        // Initialize arrays
        m_aTrackedVehicles = new array<IEntity>();
        m_aVehicleSimulations = new array<VehicleWheeledSimulation>();
        m_aLastPositions = new array<vector>();
        m_aLastSpeeds = new array<float>();
        m_aDistances = new array<float>();
        m_aAccelerations = new array<float>();
        m_aCurrentWaypointIndices = new array<int>();
        m_aWaypointTypes = new array<int>();
        m_aSpeedLimits = new array<float>();
        m_aWaypointDescriptions = new array<string>();

        m_bSessionInitialized = false;
        m_iFrameCounter = 0;
        m_bAnchorSelectorInitialized = false;
        m_iAnchorFrameCount = 0;
        m_bScreenshotCaptureInitialized = false;

        // Find or create anchor frame selector
        if (m_bEnableAnchorSelection)
        {
            m_AnchorSelector = SCR_AnchorFrameSelector.Cast(owner.FindComponent(SCR_AnchorFrameSelector));
            if (!m_AnchorSelector)
            {
                Print("[MLDataCollector] WARNING: SCR_AnchorFrameSelector not found. Add the component to enable anchor selection.", LogLevel.WARNING);
            }
        }

        // Find screenshot capture component
        if (m_bCaptureScreenshots)
        {
            m_ScreenshotCapture = SCR_MLScreenshotCapture.Cast(owner.FindComponent(SCR_MLScreenshotCapture));
            if (!m_ScreenshotCapture)
            {
                Print("[MLDataCollector] WARNING: SCR_MLScreenshotCapture not found. Add the component to enable screenshot capture.", LogLevel.WARNING);
            }
        }

        // Auto-initialize session after a short delay (allow world to load)
        GetGame().GetCallqueue().CallLater(AutoInitialize, 3000, false);

        Print("[MLDataCollector] Component initialized", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Auto-initialize session and find vehicles
    protected void AutoInitialize()
    {
        Print("[MLDataCollector] Auto-initializing session...", LogLevel.NORMAL);

        // Initialize session
        if (!InitializeSession())
        {
            Print("[MLDataCollector] ERROR: Failed to auto-initialize session", LogLevel.ERROR);
            return;
        }

        // Auto-discover vehicles in the world
        AutoDiscoverVehicles();

        Print("[MLDataCollector] Auto-initialization complete. Tracking " + m_aTrackedVehicles.Count().ToString() + " vehicles.", LogLevel.NORMAL);

        // Start periodic capture using CallLater (more reliable than EOnFrame)
        GetGame().GetCallqueue().CallLater(PeriodicCapture, m_iCaptureIntervalMs, true);
        Print("[MLDataCollector] Started periodic capture every " + m_iCaptureIntervalMs.ToString() + "ms", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Auto-discover vehicles named AI_Car_X in the world
    protected void AutoDiscoverVehicles()
    {
        BaseWorld world = GetGame().GetWorld();
        if (!world)
            return;

        // Look for AI_Car_1 through AI_Car_10
        for (int i = 1; i <= 10; i++)
        {
            string vehicleName = "AI_Car_" + i.ToString();
            IEntity vehicle = world.FindEntityByName(vehicleName);
            if (vehicle)
            {
                RegisterVehicle(vehicle);
            }
        }
    }

    //------------------------------------------------------------------------
    // Periodic capture function called by CallLater
    protected void PeriodicCapture()
    {
        if (!m_bSessionInitialized || !m_bEnableDataCapture)
            return;

        CaptureFrame();

        // Log progress periodically
        if (m_bVerboseLogging && m_iFrameCounter % m_iProgressLogInterval == 0)
        {
            Print("[MLDataCollector] Captured " + m_iFrameCounter.ToString() + " frames", LogLevel.NORMAL);
        }
    }

    //------------------------------------------------------------------------
    // Initialize a new data capture session
    bool InitializeSession()
    {
        if (!m_bEnableDataCapture)
        {
            Print("[MLDataCollector] Data capture disabled", LogLevel.WARNING);
            return false;
        }

        if (m_bSessionInitialized)
        {
            Print("[MLDataCollector] Session already initialized", LogLevel.WARNING);
            return true;
        }

        // Create base directory
        string basePath = "$profile:DrivingData";
        FileIO.MakeDirectory(basePath);

        // Generate session ID from world time
        float worldTime = GetGame().GetWorld().GetWorldTime();
        int sessionID = worldTime;

        m_sSessionPath = basePath + "/session_" + sessionID.ToString();
        FileIO.MakeDirectory(m_sSessionPath);

        // Create frames subdirectory for screenshots
        if (m_bCaptureScreenshots)
        {
            FileIO.MakeDirectory(m_sSessionPath + "/frames");
        }

        // Initialize telemetry CSV
        m_sTelemetryPath = m_sSessionPath + "/telemetry.csv";

        if (!WriteTelemetryHeader())
        {
            Print("[MLDataCollector] ERROR: Failed to create telemetry file", LogLevel.ERROR);
            return false;
        }

        m_fSessionStartTime = worldTime;
        m_bSessionInitialized = true;
        m_iFrameCounter = 0;
        m_fLastCaptureTime = 0;
        m_iAnchorFrameCount = 0;

        // Initialize anchor frame selector if enabled
        if (m_bEnableAnchorSelection && m_AnchorSelector)
        {
            if (m_AnchorSelector.InitializeSession(m_sSessionPath))
            {
                m_bAnchorSelectorInitialized = true;
                Print("[MLDataCollector] Anchor frame selection enabled", LogLevel.NORMAL);
            }
            else
            {
                Print("[MLDataCollector] WARNING: Failed to initialize anchor selector", LogLevel.WARNING);
                m_bAnchorSelectorInitialized = false;
            }
        }

        // Initialize screenshot capture if enabled
        if (m_bCaptureScreenshots && m_ScreenshotCapture)
        {
            if (m_ScreenshotCapture.InitializeSession(m_sSessionPath))
            {
                m_bScreenshotCaptureInitialized = true;
                Print("[MLDataCollector] Screenshot capture enabled", LogLevel.NORMAL);
            }
            else
            {
                Print("[MLDataCollector] WARNING: Failed to initialize screenshot capture", LogLevel.WARNING);
                m_bScreenshotCaptureInitialized = false;
            }
        }

        Print("[MLDataCollector] Session initialized: " + m_sSessionPath, LogLevel.NORMAL);
        Print("[MLDataCollector] Capture rate: " + m_iCaptureIntervalMs + "ms (" + (1000.0 / m_iCaptureIntervalMs).ToString(4, 1) + " Hz)", LogLevel.NORMAL);

        return true;
    }

    //------------------------------------------------------------------------
    // Write CSV header row
    protected bool WriteTelemetryHeader()
    {
        FileHandle file = FileIO.OpenFile(m_sTelemetryPath, FileMode.WRITE);
        if (!file)
        {
            return false;
        }

        // CSV header - matches comma.ai/world model training format
        string header = "frame_id,timestamp_ms,";
        header += "pos_x,pos_y,pos_z,";
        header += "fwd_x,fwd_y,fwd_z,";
        header += "up_x,up_y,up_z,";
        header += "right_x,right_y,right_z,";
        header += "speed_kmh,steering,throttle,brake,clutch,";
        header += "gear,engine_rpm,engine_on,handbrake,";
        header += "acceleration_kmh_s,distance_total_m,";
        header += "waypoint_type,speed_limit_kmh,";
        header += "waypoint_idx,vehicle_id";

        file.WriteLine(header);
        file.Close();

        return true;
    }

    //------------------------------------------------------------------------
    // Write session metadata file
    void WriteSessionInfo(string mapName, string weather, float timeOfDay)
    {
        if (!m_bSessionInitialized)
            return;

        string infoPath = m_sSessionPath + "/session_info.txt";
        FileHandle file = FileIO.OpenFile(infoPath, FileMode.WRITE);
        if (!file)
            return;

        file.WriteLine("=== ENFUSION DRIVING DATA CAPTURE SESSION ===");
        file.WriteLine("capture_interval_ms=" + m_iCaptureIntervalMs.ToString());
        file.WriteLine("capture_hz=" + (1000.0 / m_iCaptureIntervalMs).ToString(4, 1));
        file.WriteLine("map=" + mapName);
        file.WriteLine("weather=" + weather);
        file.WriteLine("time_of_day=" + timeOfDay.ToString(4, 2));
        file.WriteLine("screenshot_capture=" + m_bCaptureScreenshots.ToString());
        file.WriteLine("session_start_time=" + m_fSessionStartTime.ToString(12, 1));
        file.WriteLine("");

        file.WriteLine("=== TRACKED VEHICLES ===");
        for (int i = 0; i < m_aTrackedVehicles.Count(); i++)
        {
            IEntity vehicle = m_aTrackedVehicles[i];
            if (vehicle)
            {
                file.WriteLine("vehicle_" + i.ToString() + "=" + vehicle.GetName());
            }
        }

        file.WriteLine("");
        file.WriteLine("=== WAYPOINT CLASSIFICATIONS ===");
        for (int i = 0; i < m_aWaypointDescriptions.Count(); i++)
        {
            string desc = m_aWaypointDescriptions[i];
            float limit = 0;
            if (i < m_aSpeedLimits.Count())
                limit = m_aSpeedLimits[i];
            file.WriteLine("wp_" + i.ToString() + "=" + desc + " (limit: " + limit.ToString(5, 1) + " km/h)");
        }

        file.Close();
        Print("[MLDataCollector] Session info written to: " + infoPath, LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Register a vehicle for tracking
    void RegisterVehicle(IEntity vehicle, int waypointIndex = -1)
    {
        if (!vehicle)
            return;

        // Check if already registered
        int existingIndex = m_aTrackedVehicles.Find(vehicle);
        if (existingIndex >= 0)
        {
            Print("[MLDataCollector] Vehicle already registered: " + vehicle.GetName(), LogLevel.WARNING);
            return;
        }

        // Get vehicle simulation component
        VehicleWheeledSimulation sim = VehicleWheeledSimulation.Cast(vehicle.FindComponent(VehicleWheeledSimulation));
        if (!sim)
        {
            Print("[MLDataCollector] ERROR: No VehicleWheeledSimulation on " + vehicle.GetName(), LogLevel.ERROR);
            return;
        }

        // Get initial position
        vector transform[4];
        vehicle.GetWorldTransform(transform);

        // Register vehicle
        m_aTrackedVehicles.Insert(vehicle);
        m_aVehicleSimulations.Insert(sim);
        m_aLastPositions.Insert(transform[3]);
        m_aLastSpeeds.Insert(0);
        m_aDistances.Insert(0);
        m_aAccelerations.Insert(0);
        m_aCurrentWaypointIndices.Insert(waypointIndex);

        // Tell screenshot capture about the first tracked vehicle
        if (m_aTrackedVehicles.Count() == 1 && m_ScreenshotCapture)
        {
            m_ScreenshotCapture.SetTrackedVehicle(vehicle);
        }

        Print("[MLDataCollector] Registered vehicle: " + vehicle.GetName() + " (index " + (m_aTrackedVehicles.Count() - 1).ToString() + ")", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Unregister a vehicle
    void UnregisterVehicle(IEntity vehicle)
    {
        int index = m_aTrackedVehicles.Find(vehicle);
        if (index < 0)
            return;

        m_aTrackedVehicles.Remove(index);
        m_aVehicleSimulations.Remove(index);
        m_aLastPositions.Remove(index);
        m_aLastSpeeds.Remove(index);
        m_aDistances.Remove(index);
        m_aAccelerations.Remove(index);
        m_aCurrentWaypointIndices.Remove(index);

        Print("[MLDataCollector] Unregistered vehicle: " + vehicle.GetName(), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Update waypoint index for a vehicle
    void SetVehicleWaypoint(int vehicleIndex, int waypointIndex)
    {
        if (vehicleIndex >= 0 && vehicleIndex < m_aCurrentWaypointIndices.Count())
        {
            m_aCurrentWaypointIndices[vehicleIndex] = waypointIndex;
        }
    }

    //------------------------------------------------------------------------
    // Set waypoint classifications
    void SetWaypointClassifications(array<int> types, array<float> speedLimits, array<string> descriptions)
    {
        m_aWaypointTypes.Clear();
        m_aSpeedLimits.Clear();
        m_aWaypointDescriptions.Clear();

        if (types)
        {
            for (int i = 0; i < types.Count(); i++)
            {
                m_aWaypointTypes.Insert(types[i]);
            }
        }

        if (speedLimits)
        {
            for (int i = 0; i < speedLimits.Count(); i++)
            {
                m_aSpeedLimits.Insert(speedLimits[i]);
            }
        }

        if (descriptions)
        {
            for (int i = 0; i < descriptions.Count(); i++)
            {
                m_aWaypointDescriptions.Insert(descriptions[i]);
            }
        }
    }

    //------------------------------------------------------------------------
    // Main capture tick - call this from your game mode's update loop
    void CaptureFrame()
    {
        if (!m_bSessionInitialized || !m_bEnableDataCapture)
            return;

        float currentTime = GetGame().GetWorld().GetWorldTime();

        // Rate limiting
        if ((currentTime - m_fLastCaptureTime) < m_iCaptureIntervalMs)
            return;

        m_fLastCaptureTime = currentTime;

        // Capture each tracked vehicle
        for (int i = 0; i < m_aTrackedVehicles.Count(); i++)
        {
            IEntity vehicle = m_aTrackedVehicles[i];
            VehicleWheeledSimulation sim = m_aVehicleSimulations[i];

            if (!vehicle || !sim)
                continue;

            // Update telemetry calculations
            UpdateVehicleTelemetry(i, sim, vehicle);

            // Write CSV row
            CaptureVehicleState(i, sim, vehicle);

            // Process anchor frame selection for primary vehicle (first tracked)
            if (i == 0 && m_bAnchorSelectorInitialized && m_AnchorSelector)
            {
                int roadType = -1;
                int waypointIdx = m_aCurrentWaypointIndices[i];
                if (waypointIdx >= 0 && waypointIdx < m_aWaypointTypes.Count())
                    roadType = m_aWaypointTypes[waypointIdx];

                bool isAnchor = m_AnchorSelector.ProcessFrame(
                    m_iFrameCounter,
                    currentTime,
                    vehicle,
                    sim,
                    roadType
                );

                if (isAnchor)
                    m_iAnchorFrameCount++;
            }
        }

        // Capture screenshot if enabled
        if (m_bCaptureScreenshots)
        {
            CaptureScreenshot();
        }

        // Increment frame counter
        m_iFrameCounter++;

        // Progress logging
        if (m_iFrameCounter % m_iProgressLogInterval == 0)
        {
            float elapsedMin = (currentTime - m_fSessionStartTime) / 60000.0;
            Print("[MLDataCollector] " + m_iFrameCounter.ToString() + " frames captured (" + elapsedMin.ToString(6, 1) + " minutes)", LogLevel.NORMAL);
        }
    }

    //------------------------------------------------------------------------
    // Update computed telemetry values
    protected void UpdateVehicleTelemetry(int index, VehicleWheeledSimulation sim, IEntity vehicle)
    {
        // Get current state
        float speed = sim.GetSpeedKmh();

        vector transform[4];
        vehicle.GetWorldTransform(transform);
        vector currentPosition = transform[3];

        // Calculate distance traveled
        vector lastPosition = m_aLastPositions[index];
        float distance = vector.Distance(currentPosition, lastPosition);
        m_aDistances[index] = m_aDistances[index] + distance;
        m_aLastPositions[index] = currentPosition;

        // Calculate acceleration
        float lastSpeed = m_aLastSpeeds[index];
        float intervalSec = m_iCaptureIntervalMs / 1000.0;
        float acceleration = (speed - lastSpeed) / intervalSec;
        m_aAccelerations[index] = acceleration;
        m_aLastSpeeds[index] = speed;
    }

    //------------------------------------------------------------------------
    // Write a single CSV row for one vehicle
    protected void CaptureVehicleState(int vehicleIndex, VehicleWheeledSimulation sim, IEntity vehicle)
    {
        // Get timestamp
        float worldTimeMs = GetGame().GetWorld().GetWorldTime();

        // Get full 4x4 world transform
        vector transform[4];
        vehicle.GetWorldTransform(transform);

        // transform[0] = right vector (X axis)
        // transform[1] = up vector (Y axis)
        // transform[2] = forward vector (Z axis)
        // transform[3] = position
        vector position = transform[3];
        vector forward = transform[2];
        vector up = transform[1];
        vector right = transform[0];

        // Get vehicle physics state
        float speed = sim.GetSpeedKmh();
        float steering = sim.GetSteering();
        float throttle = sim.GetThrottle();
        float brake = sim.GetBrake();
        float clutch = sim.GetClutch();
        int gear = sim.GetGear();
        float engineRPM = sim.EngineGetRPM();
        bool engineOn = sim.EngineIsOn();
        bool handbrakeOn = sim.IsHandbrakeOn();

        // Get computed values
        float acceleration = m_aAccelerations[vehicleIndex];
        float distanceTotal = m_aDistances[vehicleIndex];

        // Get waypoint context
        int waypointType = -1;
        float speedLimit = 0;
        int waypointIdx = -1;

        if (vehicleIndex < m_aCurrentWaypointIndices.Count())
        {
            waypointIdx = m_aCurrentWaypointIndices[vehicleIndex];
            if (waypointIdx >= 0 && waypointIdx < m_aWaypointTypes.Count())
            {
                waypointType = m_aWaypointTypes[waypointIdx];
            }
            if (waypointIdx >= 0 && waypointIdx < m_aSpeedLimits.Count())
            {
                speedLimit = m_aSpeedLimits[waypointIdx];
            }
        }

        // Convert bools to int for CSV
        int engineOnInt = 0;
        if (engineOn)
            engineOnInt = 1;
        int handbrakeInt = 0;
        if (handbrakeOn)
            handbrakeInt = 1;

        // Build CSV row using string concatenation
        string row = m_iFrameCounter.ToString() + ",";
        row += worldTimeMs.ToString(12, 1) + ",";

        // Position (3 floats)
        row += position[0].ToString(10, 4) + ",";
        row += position[1].ToString(10, 4) + ",";
        row += position[2].ToString(10, 4) + ",";

        // Forward vector (3 floats)
        row += forward[0].ToString(8, 6) + ",";
        row += forward[1].ToString(8, 6) + ",";
        row += forward[2].ToString(8, 6) + ",";

        // Up vector (3 floats)
        row += up[0].ToString(8, 6) + ",";
        row += up[1].ToString(8, 6) + ",";
        row += up[2].ToString(8, 6) + ",";

        // Right vector (3 floats)
        row += right[0].ToString(8, 6) + ",";
        row += right[1].ToString(8, 6) + ",";
        row += right[2].ToString(8, 6) + ",";

        // Vehicle state
        row += speed.ToString(8, 3) + ",";
        row += steering.ToString(8, 5) + ",";
        row += throttle.ToString(8, 5) + ",";
        row += brake.ToString(8, 5) + ",";
        row += clutch.ToString(8, 5) + ",";

        row += gear.ToString() + ",";
        row += engineRPM.ToString(8, 1) + ",";
        row += engineOnInt.ToString() + ",";
        row += handbrakeInt.ToString() + ",";

        // Computed values
        row += acceleration.ToString(8, 3) + ",";
        row += distanceTotal.ToString(10, 2) + ",";

        // Waypoint context
        row += waypointType.ToString() + ",";
        row += speedLimit.ToString(6, 1) + ",";
        row += waypointIdx.ToString() + ",";
        row += vehicleIndex.ToString();

        // Append to CSV
        FileHandle file = FileIO.OpenFile(m_sTelemetryPath, FileMode.APPEND);
        if (file)
        {
            file.WriteLine(row);
            file.Close();
        }

        // Verbose logging
        if (m_bVerboseLogging && m_iFrameCounter % 50 == 0)
        {
            Print("[MLDataCollector] V" + vehicleIndex.ToString() + ": " + speed.ToString(5, 1) + " km/h, steer=" + steering.ToString(4, 2) + ", throttle=" + throttle.ToString(4, 2), LogLevel.VERBOSE);
        }
    }

    //------------------------------------------------------------------------
    // Capture screenshot using SCR_MLScreenshotCapture component
    protected void CaptureScreenshot()
    {
        if (m_bScreenshotCaptureInitialized && m_ScreenshotCapture)
        {
            // Delegate to the screenshot capture component
            m_ScreenshotCapture.CaptureFrame(m_iFrameCounter);
        }
        else
        {
            // Fallback: Try direct System.MakeScreenshot
            string frameNum = m_iFrameCounter.ToString();
            while (frameNum.Length() < 6)
            {
                frameNum = "0" + frameNum;
            }

            string framePath = m_sSessionPath + "/frames/frame_" + frameNum + ".bmp";

            // Ensure frames directory exists
            if (m_iFrameCounter == 0)
            {
                FileIO.MakeDirectory(m_sSessionPath + "/frames");
            }

            // Attempt direct screenshot capture
            bool success = System.MakeScreenshot(framePath);

            if (!success && m_iFrameCounter == 0)
            {
                Print("[MLDataCollector] WARNING: Screenshot capture failed. Consider adding SCR_MLScreenshotCapture component.", LogLevel.WARNING);
            }
        }
    }

    //------------------------------------------------------------------------
    // Get session statistics
    void GetSessionStats(out int frameCount, out float elapsedMinutes, out int vehicleCount, out float totalDistance)
    {
        frameCount = m_iFrameCounter;
        elapsedMinutes = (GetGame().GetWorld().GetWorldTime() - m_fSessionStartTime) / 60000.0;
        vehicleCount = m_aTrackedVehicles.Count();

        totalDistance = 0;
        for (int i = 0; i < m_aDistances.Count(); i++)
        {
            totalDistance += m_aDistances[i];
        }
    }

    //------------------------------------------------------------------------
    // Finalize session and close files
    void FinalizeSession()
    {
        if (!m_bSessionInitialized)
            return;

        int frameCount;
        float elapsedMinutes;
        int vehicleCount;
        float totalDistance;
        GetSessionStats(frameCount, elapsedMinutes, vehicleCount, totalDistance);

        // Finalize anchor selector
        if (m_bAnchorSelectorInitialized && m_AnchorSelector)
        {
            m_AnchorSelector.FinalizeSession(m_sSessionPath);
            m_bAnchorSelectorInitialized = false;
        }

        // Finalize screenshot capture
        int screenshotCount = 0;
        if (m_bScreenshotCaptureInitialized && m_ScreenshotCapture)
        {
            screenshotCount = m_ScreenshotCapture.GetFramesCaptured();
            m_ScreenshotCapture.FinalizeSession();
            m_bScreenshotCaptureInitialized = false;
        }

        // Write summary file
        string summaryPath = m_sSessionPath + "/summary.txt";
        FileHandle file = FileIO.OpenFile(summaryPath, FileMode.WRITE);
        if (file)
        {
            file.WriteLine("=== SESSION SUMMARY ===");
            file.WriteLine("total_frames=" + frameCount.ToString());
            file.WriteLine("duration_minutes=" + elapsedMinutes.ToString(8, 2));
            file.WriteLine("vehicles_tracked=" + vehicleCount.ToString());
            file.WriteLine("total_distance_m=" + totalDistance.ToString(10, 2));
            file.WriteLine("capture_hz=" + (1000.0 / m_iCaptureIntervalMs).ToString(4, 1));

            // Add anchor statistics if enabled
            if (m_bEnableAnchorSelection)
            {
                file.WriteLine("");
                file.WriteLine("=== ANCHOR FRAME STATISTICS ===");
                file.WriteLine("anchor_frames=" + m_iAnchorFrameCount.ToString());
                file.WriteLine("anchor_density_percent=" + (m_iAnchorFrameCount * 100.0 / Math.Max(1, frameCount)).ToString(6, 2));
            }

            // Add screenshot statistics if enabled
            if (m_bCaptureScreenshots && screenshotCount > 0)
            {
                file.WriteLine("");
                file.WriteLine("=== SCREENSHOT STATISTICS ===");
                file.WriteLine("screenshots_captured=" + screenshotCount.ToString());
                file.WriteLine("screenshot_capture_rate=" + (screenshotCount * 100.0 / Math.Max(1, frameCount)).ToString(6, 2) + "%");
            }

            file.Close();
        }

        Print("[MLDataCollector] Session finalized: " + frameCount.ToString() + " frames, " + elapsedMinutes.ToString(6, 1) + " minutes", LogLevel.NORMAL);
        if (m_bEnableAnchorSelection)
        {
            Print("[MLDataCollector] Anchor frames: " + m_iAnchorFrameCount.ToString() + " (" +
                  (m_iAnchorFrameCount * 100.0 / Math.Max(1, frameCount)).ToString(4, 1) + "%)", LogLevel.NORMAL);
        }
        if (m_bCaptureScreenshots && screenshotCount > 0)
        {
            Print("[MLDataCollector] Screenshots: " + screenshotCount.ToString() + " frames captured", LogLevel.NORMAL);
        }
        Print("[MLDataCollector] Data saved to: " + m_sSessionPath, LogLevel.NORMAL);

        m_bSessionInitialized = false;
    }

    //------------------------------------------------------------------------
    // Public API: Check if capturing
    bool IsCapturing()
    {
        return m_bSessionInitialized && m_bEnableDataCapture;
    }

    //------------------------------------------------------------------------
    // Public API: Get session path
    string GetSessionPath()
    {
        return m_sSessionPath;
    }

    //------------------------------------------------------------------------
    // Public API: Get frame count
    int GetFrameCount()
    {
        return m_iFrameCounter;
    }

    //------------------------------------------------------------------------
    // Public API: Get anchor frame count
    int GetAnchorFrameCount()
    {
        return m_iAnchorFrameCount;
    }

    //------------------------------------------------------------------------
    // Public API: Check if anchor selection is enabled
    bool IsAnchorSelectionEnabled()
    {
        return m_bEnableAnchorSelection && m_bAnchorSelectorInitialized;
    }

    //------------------------------------------------------------------------
    // Public API: Get anchor selector component
    SCR_AnchorFrameSelector GetAnchorSelector()
    {
        return m_AnchorSelector;
    }

    //------------------------------------------------------------------------
    // Public API: Get screenshot capture component
    SCR_MLScreenshotCapture GetScreenshotCapture()
    {
        return m_ScreenshotCapture;
    }

    //------------------------------------------------------------------------
    // Public API: Check if screenshot capture is enabled
    bool IsScreenshotCaptureEnabled()
    {
        return m_bCaptureScreenshots && m_bScreenshotCaptureInitialized;
    }

    //------------------------------------------------------------------------
    // Public API: Manually trigger an anchor frame
    void TriggerManualAnchor()
    {
        if (!m_bAnchorSelectorInitialized || !m_AnchorSelector)
        {
            Print("[MLDataCollector] Cannot trigger manual anchor - selector not initialized", LogLevel.WARNING);
            return;
        }

        float currentTime = GetGame().GetWorld().GetWorldTime();
        m_AnchorSelector.TriggerManualAnchor(currentTime);
        m_iAnchorFrameCount++;
    }

    //------------------------------------------------------------------------
    // Cleanup on component destruction
    override void OnDelete(IEntity owner)
    {
        if (m_bSessionInitialized)
        {
            FinalizeSession();
        }

        super.OnDelete(owner);
    }
}
