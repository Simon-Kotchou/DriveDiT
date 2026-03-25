// ============================================================================
// SCR_AnchorFrameSelector - Anchor Frame Selection for Self-Forcing++ Training
// ============================================================================
//
// Implements anchor frame selection for Self-Forcing++ world model training.
// Anchor frames provide ground-truth "reset points" during training, enabling
// stable generation from 5 seconds to 4+ minutes.
//
// ANCHOR SELECTION CRITERIA:
//   1. Periodic: Every N frames (configurable)
//   2. Event-triggered:
//      - Road type change (city <-> highway)
//      - Junction traversal
//      - Vehicle stop/start
//      - Significant occlusion change
//      - Large steering angle change
//   3. Quality-based:
//      - Sharp frame (not motion blurred)
//      - Good visibility (not occluded)
//
// INTEGRATION:
//   Works alongside SCR_MLDataCollector to annotate frames as anchors.
//   Outputs anchor metadata for Self-Forcing++ training pipeline.
//
// OUTPUT:
//   $profile:DrivingData/session_XXXX/anchors.csv
//   $profile:DrivingData/session_XXXX/anchor_summary.txt
//
// ============================================================================

// Anchor trigger type enumeration
enum SCR_AnchorTriggerType
{
    PERIODIC = 0,          // Regular interval anchor
    ROAD_CHANGE = 1,       // Road type transition (city <-> highway)
    JUNCTION = 2,          // Junction/intersection traversal
    STOP = 3,              // Vehicle stop event
    START = 4,             // Vehicle start after stop
    OCCLUSION = 5,         // Significant occlusion change
    STEERING = 6,          // Large steering angle change
    SPEED_CHANGE = 7,      // Significant speed change
    MANUAL = 8             // Manually triggered anchor
}

// Driving context state for state machine
enum SCR_DrivingContextState
{
    UNKNOWN = 0,
    CITY_DRIVING = 1,
    HIGHWAY_DRIVING = 2,
    RURAL_DRIVING = 3,
    OFFROAD_DRIVING = 4,
    STOPPED = 5,
    JUNCTION_APPROACH = 6,
    JUNCTION_TRAVERSAL = 7,
    JUNCTION_EXIT = 8
}

// Anchor frame metadata structure
class SCR_AnchorFrameMetadata
{
    int m_iFrameId;
    float m_fTimestampMs;
    SCR_AnchorTriggerType m_eTriggerType;
    int m_iContextWindow;           // Frames before anchor to include
    int m_iRolloutHorizon;          // Frames after anchor for self-forcing
    float m_fQualityScore;          // 0.0 to 1.0 quality assessment
    SCR_DrivingContextState m_eContextState;
    vector m_vPosition;
    float m_fSpeed;
    float m_fSteering;
    string m_sDescription;

    //------------------------------------------------------------------------
    void SCR_AnchorFrameMetadata()
    {
        m_iFrameId = 0;
        m_fTimestampMs = 0;
        m_eTriggerType = SCR_AnchorTriggerType.PERIODIC;
        m_iContextWindow = 30;      // Default 6 seconds at 5Hz
        m_iRolloutHorizon = 150;    // Default 30 seconds at 5Hz
        m_fQualityScore = 1.0;
        m_eContextState = SCR_DrivingContextState.UNKNOWN;
        m_vPosition = vector.Zero;
        m_fSpeed = 0;
        m_fSteering = 0;
        m_sDescription = "";
    }

    //------------------------------------------------------------------------
    string ToCSVRow()
    {
        string row = m_iFrameId.ToString() + ",";
        row += m_fTimestampMs.ToString(12, 1) + ",";
        row += ((int)m_eTriggerType).ToString() + ",";
        row += GetTriggerTypeName() + ",";
        row += m_iContextWindow.ToString() + ",";
        row += m_iRolloutHorizon.ToString() + ",";
        row += m_fQualityScore.ToString(6, 4) + ",";
        row += ((int)m_eContextState).ToString() + ",";
        row += GetContextStateName() + ",";
        row += m_vPosition[0].ToString(10, 4) + ",";
        row += m_vPosition[1].ToString(10, 4) + ",";
        row += m_vPosition[2].ToString(10, 4) + ",";
        row += m_fSpeed.ToString(8, 3) + ",";
        row += m_fSteering.ToString(8, 5) + ",";
        row += "\"" + m_sDescription + "\"";
        return row;
    }

    //------------------------------------------------------------------------
    string GetTriggerTypeName()
    {
        switch (m_eTriggerType)
        {
            case SCR_AnchorTriggerType.PERIODIC:      return "PERIODIC";
            case SCR_AnchorTriggerType.ROAD_CHANGE:   return "ROAD_CHANGE";
            case SCR_AnchorTriggerType.JUNCTION:      return "JUNCTION";
            case SCR_AnchorTriggerType.STOP:          return "STOP";
            case SCR_AnchorTriggerType.START:         return "START";
            case SCR_AnchorTriggerType.OCCLUSION:     return "OCCLUSION";
            case SCR_AnchorTriggerType.STEERING:      return "STEERING";
            case SCR_AnchorTriggerType.SPEED_CHANGE:  return "SPEED_CHANGE";
            case SCR_AnchorTriggerType.MANUAL:        return "MANUAL";
        }
        return "UNKNOWN";
    }

    //------------------------------------------------------------------------
    string GetContextStateName()
    {
        switch (m_eContextState)
        {
            case SCR_DrivingContextState.UNKNOWN:            return "UNKNOWN";
            case SCR_DrivingContextState.CITY_DRIVING:       return "CITY";
            case SCR_DrivingContextState.HIGHWAY_DRIVING:    return "HIGHWAY";
            case SCR_DrivingContextState.RURAL_DRIVING:      return "RURAL";
            case SCR_DrivingContextState.OFFROAD_DRIVING:    return "OFFROAD";
            case SCR_DrivingContextState.STOPPED:            return "STOPPED";
            case SCR_DrivingContextState.JUNCTION_APPROACH:  return "JUNCTION_APPROACH";
            case SCR_DrivingContextState.JUNCTION_TRAVERSAL: return "JUNCTION_TRAVERSAL";
            case SCR_DrivingContextState.JUNCTION_EXIT:      return "JUNCTION_EXIT";
        }
        return "UNKNOWN";
    }
}

// ============================================================================
// Main Anchor Frame Selector Component
// ============================================================================

[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Anchor Frame Selector for Self-Forcing++ Training")]
class SCR_AnchorFrameSelectorClass: ScriptComponentClass
{
}

class SCR_AnchorFrameSelector: ScriptComponent
{
    // === PERIODIC ANCHOR CONFIGURATION ===
    [Attribute("150", UIWidgets.Slider, "Periodic anchor interval (frames). 150 = 30 seconds at 5Hz", "30 600 15")]
    protected int m_iPeriodicAnchorInterval;

    [Attribute("1", UIWidgets.CheckBox, "Enable periodic anchors")]
    protected bool m_bEnablePeriodicAnchors;

    // === EVENT TRIGGER CONFIGURATION ===
    [Attribute("1", UIWidgets.CheckBox, "Enable road type change triggers")]
    protected bool m_bEnableRoadChangeAnchors;

    [Attribute("1", UIWidgets.CheckBox, "Enable junction traversal triggers")]
    protected bool m_bEnableJunctionAnchors;

    [Attribute("1", UIWidgets.CheckBox, "Enable stop/start triggers")]
    protected bool m_bEnableStopStartAnchors;

    [Attribute("1", UIWidgets.CheckBox, "Enable large steering change triggers")]
    protected bool m_bEnableSteeringAnchors;

    [Attribute("1", UIWidgets.CheckBox, "Enable significant speed change triggers")]
    protected bool m_bEnableSpeedChangeAnchors;

    // === THRESHOLD CONFIGURATION ===
    [Attribute("0.4", UIWidgets.Slider, "Steering change threshold (0-1 range)", "0.1 0.8 0.05")]
    protected float m_fSteeringChangeThreshold;

    [Attribute("30.0", UIWidgets.Slider, "Speed change threshold (km/h)", "10 60 5")]
    protected float m_fSpeedChangeThreshold;

    [Attribute("3.0", UIWidgets.Slider, "Stop detection speed threshold (km/h)", "0.5 10 0.5")]
    protected float m_fStopSpeedThreshold;

    [Attribute("2.0", UIWidgets.Slider, "Minimum time stopped before anchor (seconds)", "0.5 5 0.5")]
    protected float m_fMinStopDuration;

    [Attribute("15.0", UIWidgets.Slider, "Junction detection radius (meters)", "5 50 5")]
    protected float m_fJunctionDetectionRadius;

    // === QUALITY ASSESSMENT CONFIGURATION ===
    [Attribute("0.5", UIWidgets.Slider, "Minimum quality score for anchor", "0.1 1.0 0.1")]
    protected float m_fMinQualityScore;

    [Attribute("30.0", UIWidgets.Slider, "Motion blur speed threshold (km/h)", "10 80 5")]
    protected float m_fMotionBlurSpeedThreshold;

    // === CONTEXT WINDOW CONFIGURATION ===
    [Attribute("30", UIWidgets.Slider, "Default context window (frames before anchor)", "10 100 5")]
    protected int m_iDefaultContextWindow;

    [Attribute("150", UIWidgets.Slider, "Default rollout horizon (frames after anchor)", "30 300 15")]
    protected int m_iDefaultRolloutHorizon;

    // === MINIMUM ANCHOR SPACING ===
    [Attribute("25", UIWidgets.Slider, "Minimum frames between anchors", "5 100 5")]
    protected int m_iMinAnchorSpacing;

    // === LOGGING ===
    [Attribute("1", UIWidgets.CheckBox, "Verbose anchor logging")]
    protected bool m_bVerboseLogging;

    // === INTERNAL STATE ===
    // State machine
    protected SCR_DrivingContextState m_eCurrentState;
    protected SCR_DrivingContextState m_ePreviousState;
    protected float m_fStateEntryTime;

    // Frame tracking
    protected int m_iCurrentFrameId;
    protected int m_iLastAnchorFrame;
    protected int m_iFramesSinceLastAnchor;

    // Vehicle state tracking
    protected float m_fCurrentSpeed;
    protected float m_fPreviousSpeed;
    protected float m_fCurrentSteering;
    protected float m_fPreviousSteering;
    protected vector m_vCurrentPosition;
    protected vector m_vPreviousPosition;
    protected float m_fStopStartTime;
    protected bool m_bWasStopped;

    // Road context tracking
    protected int m_iCurrentRoadType;       // Uses SCR_MLDataCollector waypoint types
    protected int m_iPreviousRoadType;
    protected bool m_bNearJunction;
    protected bool m_bWasNearJunction;
    protected float m_fJunctionEntryTime;

    // Quality assessment
    protected float m_fCurrentQualityScore;
    protected ref array<float> m_aRecentSpeeds;
    protected ref array<float> m_aRecentSteerings;
    static const int QUALITY_WINDOW_SIZE = 10;

    // Anchor storage
    protected ref array<ref SCR_AnchorFrameMetadata> m_aAnchorFrames;
    protected string m_sAnchorsFilePath;
    protected bool m_bSessionInitialized;

    // Road network for junction detection
    protected SCR_AIWorld m_AIWorld;
    protected RoadNetworkManager m_RoadNetworkManager;

    // Statistics
    protected ref map<SCR_AnchorTriggerType, int> m_AnchorCounts;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        // Initialize state
        m_eCurrentState = SCR_DrivingContextState.UNKNOWN;
        m_ePreviousState = SCR_DrivingContextState.UNKNOWN;
        m_fStateEntryTime = 0;

        m_iCurrentFrameId = 0;
        m_iLastAnchorFrame = -m_iMinAnchorSpacing; // Allow immediate first anchor
        m_iFramesSinceLastAnchor = m_iMinAnchorSpacing;

        m_fCurrentSpeed = 0;
        m_fPreviousSpeed = 0;
        m_fCurrentSteering = 0;
        m_fPreviousSteering = 0;
        m_vCurrentPosition = vector.Zero;
        m_vPreviousPosition = vector.Zero;
        m_fStopStartTime = 0;
        m_bWasStopped = false;

        m_iCurrentRoadType = -1;
        m_iPreviousRoadType = -1;
        m_bNearJunction = false;
        m_bWasNearJunction = false;
        m_fJunctionEntryTime = 0;

        m_fCurrentQualityScore = 1.0;

        // Initialize arrays
        m_aRecentSpeeds = new array<float>();
        m_aRecentSteerings = new array<float>();
        m_aAnchorFrames = new array<ref SCR_AnchorFrameMetadata>();
        m_AnchorCounts = new map<SCR_AnchorTriggerType, int>();

        m_bSessionInitialized = false;

        // Initialize road network for junction detection
        InitializeRoadNetwork();

        Print("[AnchorFrameSelector] Component initialized", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void InitializeRoadNetwork()
    {
        m_AIWorld = SCR_AIWorld.Cast(GetGame().GetAIWorld());
        if (m_AIWorld)
        {
            m_RoadNetworkManager = m_AIWorld.GetRoadNetworkManager();
        }

        if (!m_RoadNetworkManager)
        {
            Print("[AnchorFrameSelector] WARNING: RoadNetworkManager not available - junction detection disabled", LogLevel.WARNING);
        }
    }

    //------------------------------------------------------------------------
    // Initialize anchor capture session
    bool InitializeSession(string sessionPath)
    {
        if (m_bSessionInitialized)
        {
            Print("[AnchorFrameSelector] Session already initialized", LogLevel.WARNING);
            return true;
        }

        m_sAnchorsFilePath = sessionPath + "/anchors.csv";

        // Write CSV header
        if (!WriteAnchorHeader())
        {
            Print("[AnchorFrameSelector] ERROR: Failed to create anchors file", LogLevel.ERROR);
            return false;
        }

        // Reset state
        m_iCurrentFrameId = 0;
        m_iLastAnchorFrame = -m_iMinAnchorSpacing;
        m_aAnchorFrames.Clear();
        m_AnchorCounts.Clear();

        m_bSessionInitialized = true;
        Print("[AnchorFrameSelector] Session initialized: " + m_sAnchorsFilePath, LogLevel.NORMAL);

        return true;
    }

    //------------------------------------------------------------------------
    protected bool WriteAnchorHeader()
    {
        FileHandle file = FileIO.OpenFile(m_sAnchorsFilePath, FileMode.WRITE);
        if (!file)
            return false;

        string header = "frame_id,timestamp_ms,trigger_type_id,trigger_type_name,";
        header += "context_window,rollout_horizon,quality_score,";
        header += "context_state_id,context_state_name,";
        header += "pos_x,pos_y,pos_z,speed_kmh,steering,description";

        file.WriteLine(header);
        file.Close();

        return true;
    }

    //------------------------------------------------------------------------
    // Main processing function - call this for each captured frame
    // Returns true if current frame is designated as an anchor
    bool ProcessFrame(int frameId, float timestampMs, IEntity vehicle, VehicleWheeledSimulation vehicleSim, int roadType = -1)
    {
        if (!m_bSessionInitialized)
            return false;

        m_iCurrentFrameId = frameId;
        m_iFramesSinceLastAnchor++;

        // Update vehicle state
        UpdateVehicleState(vehicle, vehicleSim);

        // Update road context
        if (roadType >= 0)
        {
            m_iPreviousRoadType = m_iCurrentRoadType;
            m_iCurrentRoadType = roadType;
        }

        // Update junction proximity
        UpdateJunctionProximity();

        // Update driving state machine
        UpdateStateMachine(timestampMs);

        // Update quality assessment
        UpdateQualityAssessment();

        // Check for anchor triggers
        SCR_AnchorTriggerType triggerType;
        bool shouldAnchor = EvaluateAnchorTriggers(timestampMs, triggerType);

        if (shouldAnchor)
        {
            CreateAnchor(frameId, timestampMs, triggerType);
            return true;
        }

        return false;
    }

    //------------------------------------------------------------------------
    protected void UpdateVehicleState(IEntity vehicle, VehicleWheeledSimulation vehicleSim)
    {
        if (!vehicle || !vehicleSim)
            return;

        // Store previous values
        m_fPreviousSpeed = m_fCurrentSpeed;
        m_fPreviousSteering = m_fCurrentSteering;
        m_vPreviousPosition = m_vCurrentPosition;

        // Get current values
        m_fCurrentSpeed = vehicleSim.GetSpeedKmh();
        m_fCurrentSteering = vehicleSim.GetSteering();

        vector transform[4];
        vehicle.GetWorldTransform(transform);
        m_vCurrentPosition = transform[3];

        // Update quality tracking arrays
        m_aRecentSpeeds.Insert(m_fCurrentSpeed);
        m_aRecentSteerings.Insert(m_fCurrentSteering);

        // Keep arrays bounded
        while (m_aRecentSpeeds.Count() > QUALITY_WINDOW_SIZE)
            m_aRecentSpeeds.Remove(0);
        while (m_aRecentSteerings.Count() > QUALITY_WINDOW_SIZE)
            m_aRecentSteerings.Remove(0);
    }

    //------------------------------------------------------------------------
    protected void UpdateJunctionProximity()
    {
        m_bWasNearJunction = m_bNearJunction;
        m_bNearJunction = false;

        if (!m_RoadNetworkManager)
            return;

        // Get roads around current position
        array<BaseRoad> nearbyRoads = {};
        vector min = m_vCurrentPosition - Vector(m_fJunctionDetectionRadius, m_fJunctionDetectionRadius, m_fJunctionDetectionRadius);
        vector max = m_vCurrentPosition + Vector(m_fJunctionDetectionRadius, m_fJunctionDetectionRadius, m_fJunctionDetectionRadius);

        m_RoadNetworkManager.GetRoadsInAABB(min, max, nearbyRoads);

        // Junction is detected when multiple roads meet nearby
        // Simple heuristic: count road endpoints near our position
        int nearbyEndpoints = 0;
        float endpointRadius = m_fJunctionDetectionRadius * 0.5;

        foreach (BaseRoad road : nearbyRoads)
        {
            array<vector> roadPoints = {};
            road.GetPoints(roadPoints);

            if (roadPoints.IsEmpty())
                continue;

            // Check start point
            if (vector.Distance(m_vCurrentPosition, roadPoints[0]) < endpointRadius)
                nearbyEndpoints++;

            // Check end point
            if (roadPoints.Count() > 1)
            {
                if (vector.Distance(m_vCurrentPosition, roadPoints[roadPoints.Count() - 1]) < endpointRadius)
                    nearbyEndpoints++;
            }
        }

        // 3+ endpoints nearby indicates a junction
        m_bNearJunction = (nearbyEndpoints >= 3);
    }

    //------------------------------------------------------------------------
    protected void UpdateStateMachine(float timestampMs)
    {
        SCR_DrivingContextState newState = m_eCurrentState;

        // Stopped state detection
        if (m_fCurrentSpeed < m_fStopSpeedThreshold)
        {
            if (m_eCurrentState != SCR_DrivingContextState.STOPPED)
            {
                // Just stopped
                m_fStopStartTime = timestampMs;
            }
            newState = SCR_DrivingContextState.STOPPED;
        }
        // Junction state detection
        else if (m_bNearJunction)
        {
            if (!m_bWasNearJunction)
            {
                // Entering junction
                newState = SCR_DrivingContextState.JUNCTION_APPROACH;
                m_fJunctionEntryTime = timestampMs;
            }
            else if (m_eCurrentState == SCR_DrivingContextState.JUNCTION_APPROACH)
            {
                // In junction for a while -> traversing
                if ((timestampMs - m_fJunctionEntryTime) > 1000.0) // 1 second
                    newState = SCR_DrivingContextState.JUNCTION_TRAVERSAL;
            }
        }
        else if (m_bWasNearJunction && !m_bNearJunction)
        {
            // Exiting junction
            newState = SCR_DrivingContextState.JUNCTION_EXIT;
        }
        // Road type state detection
        else
        {
            switch (m_iCurrentRoadType)
            {
                case 0: // CITY
                    newState = SCR_DrivingContextState.CITY_DRIVING;
                    break;
                case 1: // HIGHWAY
                    newState = SCR_DrivingContextState.HIGHWAY_DRIVING;
                    break;
                case 2: // RURAL
                    newState = SCR_DrivingContextState.RURAL_DRIVING;
                    break;
                case 3: // OFFROAD
                    newState = SCR_DrivingContextState.OFFROAD_DRIVING;
                    break;
                default:
                    // Keep current state if road type unknown
                    if (m_eCurrentState == SCR_DrivingContextState.STOPPED ||
                        m_eCurrentState == SCR_DrivingContextState.JUNCTION_APPROACH ||
                        m_eCurrentState == SCR_DrivingContextState.JUNCTION_TRAVERSAL ||
                        m_eCurrentState == SCR_DrivingContextState.JUNCTION_EXIT)
                    {
                        // Transition back to driving
                        newState = SCR_DrivingContextState.UNKNOWN;
                    }
                    break;
            }
        }

        // State transition
        if (newState != m_eCurrentState)
        {
            m_ePreviousState = m_eCurrentState;
            m_eCurrentState = newState;
            m_fStateEntryTime = timestampMs;

            if (m_bVerboseLogging)
            {
                ref SCR_AnchorFrameMetadata tempMeta = new SCR_AnchorFrameMetadata();
                tempMeta.m_eContextState = m_ePreviousState;
                string prevName = tempMeta.GetContextStateName();
                tempMeta.m_eContextState = m_eCurrentState;
                string currName = tempMeta.GetContextStateName();
                Print("[AnchorFrameSelector] State transition: " + prevName + " -> " + currName, LogLevel.VERBOSE);
            }
        }

        // Update stopped flag
        m_bWasStopped = (m_ePreviousState == SCR_DrivingContextState.STOPPED);
    }

    //------------------------------------------------------------------------
    protected void UpdateQualityAssessment()
    {
        // Quality score based on:
        // 1. Motion blur (lower speed = better)
        // 2. Steering stability (less variation = better)
        // 3. Speed stability (less variation = better)

        float qualityScore = 1.0;

        // Motion blur penalty
        if (m_fCurrentSpeed > m_fMotionBlurSpeedThreshold)
        {
            float blurFactor = (m_fCurrentSpeed - m_fMotionBlurSpeedThreshold) / m_fMotionBlurSpeedThreshold;
            qualityScore -= Math.Clamp(blurFactor * 0.3, 0, 0.3);
        }

        // Steering stability penalty
        if (m_aRecentSteerings.Count() >= 3)
        {
            float steeringVariance = CalculateVariance(m_aRecentSteerings);
            qualityScore -= Math.Clamp(steeringVariance * 2.0, 0, 0.3);
        }

        // Speed stability penalty
        if (m_aRecentSpeeds.Count() >= 3)
        {
            float speedVariance = CalculateVariance(m_aRecentSpeeds);
            float normalizedVariance = speedVariance / (m_fSpeedChangeThreshold * m_fSpeedChangeThreshold);
            qualityScore -= Math.Clamp(normalizedVariance * 0.2, 0, 0.2);
        }

        // Bonus for stopped vehicle (very stable)
        if (m_eCurrentState == SCR_DrivingContextState.STOPPED)
        {
            qualityScore = Math.Min(qualityScore + 0.2, 1.0);
        }

        m_fCurrentQualityScore = Math.Clamp(qualityScore, 0.0, 1.0);
    }

    //------------------------------------------------------------------------
    protected float CalculateVariance(array<float> values)
    {
        if (values.Count() < 2)
            return 0;

        // Calculate mean
        float sum = 0;
        foreach (float val : values)
            sum += val;
        float mean = sum / values.Count();

        // Calculate variance
        float variance = 0;
        foreach (float val : values)
        {
            float diff = val - mean;
            variance += diff * diff;
        }
        variance /= values.Count();

        return variance;
    }

    //------------------------------------------------------------------------
    protected bool EvaluateAnchorTriggers(float timestampMs, out SCR_AnchorTriggerType triggerType)
    {
        triggerType = SCR_AnchorTriggerType.PERIODIC;

        // Enforce minimum anchor spacing
        if (m_iFramesSinceLastAnchor < m_iMinAnchorSpacing)
            return false;

        // Quality gate
        if (m_fCurrentQualityScore < m_fMinQualityScore)
            return false;

        // Check triggers in priority order

        // 1. Stop event (high priority for inverse dynamics)
        if (m_bEnableStopStartAnchors && m_eCurrentState == SCR_DrivingContextState.STOPPED)
        {
            float stopDuration = (timestampMs - m_fStopStartTime) / 1000.0;
            if (stopDuration >= m_fMinStopDuration && !m_bWasStopped)
            {
                triggerType = SCR_AnchorTriggerType.STOP;
                return true;
            }
        }

        // 2. Start after stop (high priority)
        if (m_bEnableStopStartAnchors && m_bWasStopped && m_eCurrentState != SCR_DrivingContextState.STOPPED)
        {
            triggerType = SCR_AnchorTriggerType.START;
            return true;
        }

        // 3. Road type change
        if (m_bEnableRoadChangeAnchors && m_iCurrentRoadType >= 0 && m_iPreviousRoadType >= 0)
        {
            if (m_iCurrentRoadType != m_iPreviousRoadType)
            {
                triggerType = SCR_AnchorTriggerType.ROAD_CHANGE;
                return true;
            }
        }

        // 4. Junction traversal
        if (m_bEnableJunctionAnchors)
        {
            // Anchor at junction approach
            if (m_eCurrentState == SCR_DrivingContextState.JUNCTION_APPROACH &&
                m_ePreviousState != SCR_DrivingContextState.JUNCTION_APPROACH)
            {
                triggerType = SCR_AnchorTriggerType.JUNCTION;
                return true;
            }
            // Also anchor at junction exit
            if (m_eCurrentState == SCR_DrivingContextState.JUNCTION_EXIT &&
                m_ePreviousState == SCR_DrivingContextState.JUNCTION_TRAVERSAL)
            {
                triggerType = SCR_AnchorTriggerType.JUNCTION;
                return true;
            }
        }

        // 5. Large steering change
        if (m_bEnableSteeringAnchors)
        {
            float steeringChange = Math.AbsFloat(m_fCurrentSteering - m_fPreviousSteering);
            if (steeringChange >= m_fSteeringChangeThreshold)
            {
                triggerType = SCR_AnchorTriggerType.STEERING;
                return true;
            }
        }

        // 6. Significant speed change
        if (m_bEnableSpeedChangeAnchors)
        {
            float speedChange = Math.AbsFloat(m_fCurrentSpeed - m_fPreviousSpeed);
            if (speedChange >= m_fSpeedChangeThreshold)
            {
                triggerType = SCR_AnchorTriggerType.SPEED_CHANGE;
                return true;
            }
        }

        // 7. Periodic anchor (lowest priority)
        if (m_bEnablePeriodicAnchors)
        {
            if (m_iFramesSinceLastAnchor >= m_iPeriodicAnchorInterval)
            {
                triggerType = SCR_AnchorTriggerType.PERIODIC;
                return true;
            }
        }

        return false;
    }

    //------------------------------------------------------------------------
    protected void CreateAnchor(int frameId, float timestampMs, SCR_AnchorTriggerType triggerType)
    {
        // Create anchor metadata
        ref SCR_AnchorFrameMetadata anchor = new SCR_AnchorFrameMetadata();

        anchor.m_iFrameId = frameId;
        anchor.m_fTimestampMs = timestampMs;
        anchor.m_eTriggerType = triggerType;
        anchor.m_eContextState = m_eCurrentState;
        anchor.m_vPosition = m_vCurrentPosition;
        anchor.m_fSpeed = m_fCurrentSpeed;
        anchor.m_fSteering = m_fCurrentSteering;
        anchor.m_fQualityScore = m_fCurrentQualityScore;

        // Set context window and rollout horizon based on trigger type
        SetAnchorHorizons(anchor);

        // Generate description
        anchor.m_sDescription = GenerateAnchorDescription(triggerType);

        // Store anchor
        m_aAnchorFrames.Insert(anchor);

        // Update statistics
        int count = 0;
        if (m_AnchorCounts.Contains(triggerType))
            count = m_AnchorCounts[triggerType];
        m_AnchorCounts[triggerType] = count + 1;

        // Write to file
        WriteAnchorToFile(anchor);

        // Update tracking
        m_iLastAnchorFrame = frameId;
        m_iFramesSinceLastAnchor = 0;

        if (m_bVerboseLogging)
        {
            Print("[AnchorFrameSelector] ANCHOR frame=" + frameId.ToString() +
                  " type=" + anchor.GetTriggerTypeName() +
                  " quality=" + m_fCurrentQualityScore.ToString(4, 2) +
                  " | " + anchor.m_sDescription, LogLevel.NORMAL);
        }
    }

    //------------------------------------------------------------------------
    protected void SetAnchorHorizons(SCR_AnchorFrameMetadata anchor)
    {
        // Adjust context window and rollout horizon based on trigger type
        // These values are optimized for Self-Forcing++ training

        switch (anchor.m_eTriggerType)
        {
            case SCR_AnchorTriggerType.STOP:
                // Stop events: shorter context, longer rollout
                // Good for learning start/stop dynamics
                anchor.m_iContextWindow = 15;   // 3 seconds before stop
                anchor.m_iRolloutHorizon = 200; // 40 seconds after (to capture restart)
                break;

            case SCR_AnchorTriggerType.START:
                // Start events: longer context (to see stopped state), medium rollout
                anchor.m_iContextWindow = 50;   // 10 seconds (include stop period)
                anchor.m_iRolloutHorizon = 150; // 30 seconds
                break;

            case SCR_AnchorTriggerType.JUNCTION:
                // Junction events: medium context, medium rollout
                anchor.m_iContextWindow = 25;   // 5 seconds approach
                anchor.m_iRolloutHorizon = 100; // 20 seconds (cover junction traversal)
                break;

            case SCR_AnchorTriggerType.ROAD_CHANGE:
                // Road change: longer context to show transition
                anchor.m_iContextWindow = 40;   // 8 seconds
                anchor.m_iRolloutHorizon = 150; // 30 seconds
                break;

            case SCR_AnchorTriggerType.STEERING:
                // Steering events: short context, medium rollout
                anchor.m_iContextWindow = 20;   // 4 seconds
                anchor.m_iRolloutHorizon = 100; // 20 seconds
                break;

            case SCR_AnchorTriggerType.SPEED_CHANGE:
                // Speed change: medium context and rollout
                anchor.m_iContextWindow = 25;   // 5 seconds
                anchor.m_iRolloutHorizon = 125; // 25 seconds
                break;

            case SCR_AnchorTriggerType.PERIODIC:
            default:
                // Periodic: use defaults
                anchor.m_iContextWindow = m_iDefaultContextWindow;
                anchor.m_iRolloutHorizon = m_iDefaultRolloutHorizon;
                break;
        }

        // Ensure context window doesn't exceed available frames
        if (anchor.m_iContextWindow > anchor.m_iFrameId)
            anchor.m_iContextWindow = anchor.m_iFrameId;
    }

    //------------------------------------------------------------------------
    protected string GenerateAnchorDescription(SCR_AnchorTriggerType triggerType)
    {
        string desc = "";

        switch (triggerType)
        {
            case SCR_AnchorTriggerType.PERIODIC:
                desc = "Periodic anchor at " + m_fCurrentSpeed.ToString(5, 1) + " km/h";
                break;

            case SCR_AnchorTriggerType.ROAD_CHANGE:
                desc = "Road type changed from " + GetRoadTypeName(m_iPreviousRoadType) +
                       " to " + GetRoadTypeName(m_iCurrentRoadType);
                break;

            case SCR_AnchorTriggerType.JUNCTION:
                if (m_eCurrentState == SCR_DrivingContextState.JUNCTION_APPROACH)
                    desc = "Approaching junction at " + m_fCurrentSpeed.ToString(5, 1) + " km/h";
                else
                    desc = "Exiting junction at " + m_fCurrentSpeed.ToString(5, 1) + " km/h";
                break;

            case SCR_AnchorTriggerType.STOP:
                desc = "Vehicle stopped";
                break;

            case SCR_AnchorTriggerType.START:
                desc = "Vehicle started moving at " + m_fCurrentSpeed.ToString(5, 1) + " km/h";
                break;

            case SCR_AnchorTriggerType.STEERING:
                float steerChange = m_fCurrentSteering - m_fPreviousSteering;
                string direction;
                if (steerChange > 0)
                    direction = "right";
                else
                    direction = "left";
                desc = "Large steering change (" + direction + ") delta=" +
                       Math.AbsFloat(steerChange).ToString(4, 2);
                break;

            case SCR_AnchorTriggerType.SPEED_CHANGE:
                float speedChange = m_fCurrentSpeed - m_fPreviousSpeed;
                string accelType;
                if (speedChange > 0)
                    accelType = "acceleration";
                else
                    accelType = "deceleration";
                desc = "Significant " + accelType + " delta=" +
                       Math.AbsFloat(speedChange).ToString(5, 1) + " km/h";
                break;

            case SCR_AnchorTriggerType.MANUAL:
                desc = "Manually triggered anchor";
                break;

            default:
                desc = "Anchor frame";
                break;
        }

        return desc;
    }

    //------------------------------------------------------------------------
    protected string GetRoadTypeName(int roadType)
    {
        string name = "Unknown";
        if (roadType == 0)
            name = "City";
        else if (roadType == 1)
            name = "Highway";
        else if (roadType == 2)
            name = "Rural";
        else if (roadType == 3)
            name = "Offroad";
        return name;
    }

    //------------------------------------------------------------------------
    protected void WriteAnchorToFile(SCR_AnchorFrameMetadata anchor)
    {
        FileHandle file = FileIO.OpenFile(m_sAnchorsFilePath, FileMode.APPEND);
        if (file)
        {
            file.WriteLine(anchor.ToCSVRow());
            file.Close();
        }
    }

    //------------------------------------------------------------------------
    // Manual anchor trigger (for external components)
    void TriggerManualAnchor(float timestampMs)
    {
        if (!m_bSessionInitialized)
            return;

        if (m_iFramesSinceLastAnchor < m_iMinAnchorSpacing)
        {
            Print("[AnchorFrameSelector] Manual anchor rejected - too close to previous anchor", LogLevel.WARNING);
            return;
        }

        CreateAnchor(m_iCurrentFrameId, timestampMs, SCR_AnchorTriggerType.MANUAL);
    }

    //------------------------------------------------------------------------
    // Get total anchor count
    int GetAnchorCount()
    {
        return m_aAnchorFrames.Count();
    }

    //------------------------------------------------------------------------
    // Get anchors by type
    int GetAnchorCountByType(SCR_AnchorTriggerType triggerType)
    {
        if (m_AnchorCounts.Contains(triggerType))
            return m_AnchorCounts[triggerType];
        return 0;
    }

    //------------------------------------------------------------------------
    // Get all anchor frames
    array<ref SCR_AnchorFrameMetadata> GetAnchorFrames()
    {
        return m_aAnchorFrames;
    }

    //------------------------------------------------------------------------
    // Get current quality score
    float GetCurrentQualityScore()
    {
        return m_fCurrentQualityScore;
    }

    //------------------------------------------------------------------------
    // Get current driving state
    SCR_DrivingContextState GetCurrentState()
    {
        return m_eCurrentState;
    }

    //------------------------------------------------------------------------
    // Check if near junction
    bool IsNearJunction()
    {
        return m_bNearJunction;
    }

    //------------------------------------------------------------------------
    // Finalize session and write summary
    void FinalizeSession(string sessionPath)
    {
        if (!m_bSessionInitialized)
            return;

        // Write summary
        string summaryPath = sessionPath + "/anchor_summary.txt";
        FileHandle file = FileIO.OpenFile(summaryPath, FileMode.WRITE);
        if (file)
        {
            file.WriteLine("=== ANCHOR FRAME SELECTION SUMMARY ===");
            file.WriteLine("");
            file.WriteLine("Total anchors: " + m_aAnchorFrames.Count().ToString());
            file.WriteLine("Total frames processed: " + m_iCurrentFrameId.ToString());
            file.WriteLine("Anchor density: " + (m_aAnchorFrames.Count() * 100.0 / Math.Max(1, m_iCurrentFrameId)).ToString(6, 2) + "%");
            file.WriteLine("");

            file.WriteLine("=== ANCHOR COUNTS BY TYPE ===");

            // Report counts for each trigger type
            array<SCR_AnchorTriggerType> types = {
                SCR_AnchorTriggerType.PERIODIC,
                SCR_AnchorTriggerType.ROAD_CHANGE,
                SCR_AnchorTriggerType.JUNCTION,
                SCR_AnchorTriggerType.STOP,
                SCR_AnchorTriggerType.START,
                SCR_AnchorTriggerType.OCCLUSION,
                SCR_AnchorTriggerType.STEERING,
                SCR_AnchorTriggerType.SPEED_CHANGE,
                SCR_AnchorTriggerType.MANUAL
            };

            array<string> typeNames = {
                "PERIODIC", "ROAD_CHANGE", "JUNCTION", "STOP", "START",
                "OCCLUSION", "STEERING", "SPEED_CHANGE", "MANUAL"
            };

            for (int i = 0; i < types.Count(); i++)
            {
                int count = GetAnchorCountByType(types[i]);
                if (count > 0)
                {
                    file.WriteLine(typeNames[i] + ": " + count.ToString());
                }
            }

            file.WriteLine("");
            file.WriteLine("=== CONFIGURATION ===");
            file.WriteLine("Periodic interval: " + m_iPeriodicAnchorInterval.ToString() + " frames");
            file.WriteLine("Min anchor spacing: " + m_iMinAnchorSpacing.ToString() + " frames");
            file.WriteLine("Min quality score: " + m_fMinQualityScore.ToString(4, 2));
            file.WriteLine("Steering threshold: " + m_fSteeringChangeThreshold.ToString(4, 2));
            file.WriteLine("Speed change threshold: " + m_fSpeedChangeThreshold.ToString(5, 1) + " km/h");
            file.WriteLine("Junction radius: " + m_fJunctionDetectionRadius.ToString(5, 1) + " m");

            file.Close();

            Print("[AnchorFrameSelector] Summary written to: " + summaryPath, LogLevel.NORMAL);
        }

        Print("[AnchorFrameSelector] Session finalized: " + m_aAnchorFrames.Count().ToString() + " anchors from " +
              m_iCurrentFrameId.ToString() + " frames", LogLevel.NORMAL);

        m_bSessionInitialized = false;
    }

    //------------------------------------------------------------------------
    // Reset state (for new session without destroying component)
    void Reset()
    {
        m_eCurrentState = SCR_DrivingContextState.UNKNOWN;
        m_ePreviousState = SCR_DrivingContextState.UNKNOWN;
        m_iCurrentFrameId = 0;
        m_iLastAnchorFrame = -m_iMinAnchorSpacing;
        m_iFramesSinceLastAnchor = m_iMinAnchorSpacing;

        m_fCurrentSpeed = 0;
        m_fPreviousSpeed = 0;
        m_fCurrentSteering = 0;
        m_fPreviousSteering = 0;
        m_bWasStopped = false;

        m_iCurrentRoadType = -1;
        m_iPreviousRoadType = -1;
        m_bNearJunction = false;
        m_bWasNearJunction = false;

        m_aRecentSpeeds.Clear();
        m_aRecentSteerings.Clear();
        m_aAnchorFrames.Clear();
        m_AnchorCounts.Clear();

        m_bSessionInitialized = false;

        Print("[AnchorFrameSelector] Component reset", LogLevel.NORMAL);
    }
}
