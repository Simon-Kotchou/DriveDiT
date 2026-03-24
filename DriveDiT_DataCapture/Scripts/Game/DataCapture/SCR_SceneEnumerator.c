// ============================================================================
// SCR_SceneEnumerator - Scene Graph Capture System for World Model Training
// ============================================================================
//
// Captures structured scene graph data compatible with GAIA-2 conditioning.
// Enumerates nearby entities, classifies semantically, extracts 3D bounding
// boxes (AABB), and computes velocities from position deltas.
//
// OUTPUT FORMAT:
//   Scene graph with per-entity records containing:
//   - Semantic class (vehicle, character, static, dynamic prop)
//   - 3D position (world coordinates)
//   - 3D velocity (m/s)
//   - AABB bounding box (mins, maxs in local space)
//   - Orientation (forward, up vectors)
//   - Entity ID for tracking across frames
//
// USAGE:
//   Attach to ego vehicle or scene manager entity.
//   Call CaptureSceneGraph() each frame or at desired frequency.
//
// ============================================================================

// -----------------------------------------------------------------------------
// Semantic Class Enumeration (GAIA-2 compatible)
// -----------------------------------------------------------------------------
enum ESceneEntityClass
{
    UNKNOWN = 0,
    EGO_VEHICLE = 1,
    OTHER_VEHICLE = 2,
    PEDESTRIAN = 3,
    STATIC_OBJECT = 4,
    DYNAMIC_PROP = 5,
    BUILDING = 6,
    VEGETATION = 7,
    ROAD_INFRASTRUCTURE = 8
}

// -----------------------------------------------------------------------------
// Scene Entity Data Structure
// Holds all extracted information for a single entity in the scene graph
// -----------------------------------------------------------------------------
class SCR_SceneEntityData
{
    // Identification
    int m_iEntityID;                    // Unique ID for tracking
    string m_sEntityName;               // Entity name
    ESceneEntityClass m_eSemanticClass; // Semantic classification

    // Spatial properties
    vector m_vPosition;                 // World position (center)
    vector m_vForward;                  // Forward direction vector
    vector m_vUp;                       // Up direction vector
    vector m_vRight;                    // Right direction vector

    // Bounding box (local space AABB)
    vector m_vBoundsMin;                // AABB minimum corner
    vector m_vBoundsMax;                // AABB maximum corner
    vector m_vBoundsCenter;             // AABB center (local)
    vector m_vBoundsExtents;            // AABB half-extents

    // Dynamics
    vector m_vVelocity;                 // Linear velocity (m/s)
    float m_fSpeed;                     // Speed magnitude (m/s)
    vector m_vAngularVelocity;          // Angular velocity (rad/s)

    // Distance from ego
    float m_fDistanceToEgo;             // Distance to ego vehicle

    // Tracking state
    bool m_bIsTracked;                  // Currently being tracked
    int m_iFramesSinceLastSeen;         // Frames since last observation

    //------------------------------------------------------------------------
    void SCR_SceneEntityData()
    {
        m_iEntityID = -1;
        m_sEntityName = "";
        m_eSemanticClass = ESceneEntityClass.UNKNOWN;
        m_vPosition = vector.Zero;
        m_vForward = vector.Forward;
        m_vUp = vector.Up;
        m_vRight = "1 0 0";
        m_vBoundsMin = vector.Zero;
        m_vBoundsMax = vector.Zero;
        m_vBoundsCenter = vector.Zero;
        m_vBoundsExtents = vector.Zero;
        m_vVelocity = vector.Zero;
        m_fSpeed = 0;
        m_vAngularVelocity = vector.Zero;
        m_fDistanceToEgo = 0;
        m_bIsTracked = false;
        m_iFramesSinceLastSeen = 0;
    }

    //------------------------------------------------------------------------
    // Serialize to CSV row format
    string ToCSVRow()
    {
        string row = "";

        // Entity identification
        row += m_iEntityID.ToString() + ",";
        row += m_eSemanticClass.ToString() + ",";

        // Position (3 floats)
        row += m_vPosition[0].ToString(10, 4) + ",";
        row += m_vPosition[1].ToString(10, 4) + ",";
        row += m_vPosition[2].ToString(10, 4) + ",";

        // Forward vector (3 floats)
        row += m_vForward[0].ToString(8, 6) + ",";
        row += m_vForward[1].ToString(8, 6) + ",";
        row += m_vForward[2].ToString(8, 6) + ",";

        // Bounding box min (3 floats)
        row += m_vBoundsMin[0].ToString(8, 4) + ",";
        row += m_vBoundsMin[1].ToString(8, 4) + ",";
        row += m_vBoundsMin[2].ToString(8, 4) + ",";

        // Bounding box max (3 floats)
        row += m_vBoundsMax[0].ToString(8, 4) + ",";
        row += m_vBoundsMax[1].ToString(8, 4) + ",";
        row += m_vBoundsMax[2].ToString(8, 4) + ",";

        // Velocity (3 floats)
        row += m_vVelocity[0].ToString(8, 4) + ",";
        row += m_vVelocity[1].ToString(8, 4) + ",";
        row += m_vVelocity[2].ToString(8, 4) + ",";

        // Speed and distance
        row += m_fSpeed.ToString(8, 3) + ",";
        row += m_fDistanceToEgo.ToString(8, 2);

        return row;
    }

    //------------------------------------------------------------------------
    // Get semantic class name string
    static string GetClassName(ESceneEntityClass eClass)
    {
        switch (eClass)
        {
            case ESceneEntityClass.UNKNOWN: return "unknown";
            case ESceneEntityClass.EGO_VEHICLE: return "ego_vehicle";
            case ESceneEntityClass.OTHER_VEHICLE: return "other_vehicle";
            case ESceneEntityClass.PEDESTRIAN: return "pedestrian";
            case ESceneEntityClass.STATIC_OBJECT: return "static_object";
            case ESceneEntityClass.DYNAMIC_PROP: return "dynamic_prop";
            case ESceneEntityClass.BUILDING: return "building";
            case ESceneEntityClass.VEGETATION: return "vegetation";
            case ESceneEntityClass.ROAD_INFRASTRUCTURE: return "road_infra";
            default: return "unknown";
        }
    }
}

// -----------------------------------------------------------------------------
// Scene Graph Container
// Holds the complete scene state for one capture frame
// -----------------------------------------------------------------------------
class SCR_SceneGraph
{
    // Frame metadata
    int m_iFrameID;
    float m_fTimestamp;

    // Ego vehicle state
    ref SCR_SceneEntityData m_EgoVehicle;

    // All detected entities
    ref array<ref SCR_SceneEntityData> m_aEntities;

    // Statistics
    int m_iVehicleCount;
    int m_iPedestrianCount;
    int m_iStaticCount;
    int m_iDynamicCount;

    //------------------------------------------------------------------------
    void SCR_SceneGraph()
    {
        m_iFrameID = 0;
        m_fTimestamp = 0;
        m_EgoVehicle = null;
        m_aEntities = new array<ref SCR_SceneEntityData>();
        m_iVehicleCount = 0;
        m_iPedestrianCount = 0;
        m_iStaticCount = 0;
        m_iDynamicCount = 0;
    }

    //------------------------------------------------------------------------
    void Clear()
    {
        m_aEntities.Clear();
        m_EgoVehicle = null;
        m_iVehicleCount = 0;
        m_iPedestrianCount = 0;
        m_iStaticCount = 0;
        m_iDynamicCount = 0;
    }

    //------------------------------------------------------------------------
    int GetTotalEntityCount()
    {
        return m_aEntities.Count();
    }
}

// -----------------------------------------------------------------------------
// Main Scene Enumerator Component
// -----------------------------------------------------------------------------
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Scene Graph Capture - Enumerates and classifies nearby entities for world model training")]
class SCR_SceneEnumeratorClass : ScriptComponentClass
{
}

class SCR_SceneEnumerator : ScriptComponent
{
    // === CONFIGURATION ===
    [Attribute("100", UIWidgets.Slider, "Query radius around ego vehicle (meters)", "10 500 10")]
    protected float m_fQueryRadius;

    [Attribute("50", UIWidgets.Slider, "Maximum entities to track", "10 200 10")]
    protected int m_iMaxEntities;

    [Attribute("1", UIWidgets.CheckBox, "Include vehicles in scene graph")]
    protected bool m_bIncludeVehicles;

    [Attribute("1", UIWidgets.CheckBox, "Include characters/pedestrians")]
    protected bool m_bIncludeCharacters;

    [Attribute("1", UIWidgets.CheckBox, "Include static objects")]
    protected bool m_bIncludeStaticObjects;

    [Attribute("1", UIWidgets.CheckBox, "Include dynamic props")]
    protected bool m_bIncludeDynamicProps;

    [Attribute("5.0", UIWidgets.Slider, "Minimum size for static objects (meters)", "0.5 20.0 0.5")]
    protected float m_fMinStaticSize;

    [Attribute("1", UIWidgets.CheckBox, "Enable velocity tracking")]
    protected bool m_bTrackVelocity;

    [Attribute("1", UIWidgets.CheckBox, "Enable CSV output")]
    protected bool m_bEnableCSVOutput;

    [Attribute("1", UIWidgets.CheckBox, "Verbose logging")]
    protected bool m_bVerboseLogging;

    // === STATE ===
    protected IEntity m_EgoVehicle;
    protected ref SCR_SceneGraph m_CurrentSceneGraph;
    protected int m_iFrameCounter;
    protected float m_fLastCaptureTime;

    // === VELOCITY TRACKING ===
    // Maps entity pointer hash to previous position for velocity calculation
    protected ref map<int, vector> m_mPreviousPositions;
    protected ref map<int, float> m_mPreviousTimestamps;

    // === ENTITY ID TRACKING ===
    // Persistent ID assignment for entities
    protected ref map<int, int> m_mEntityIDs;
    protected int m_iNextEntityID;

    // === CSV OUTPUT ===
    protected string m_sOutputPath;
    protected bool m_bOutputInitialized;

    // === QUERY CALLBACK STATE ===
    // Used during QueryEntitiesBySphere callback
    protected ref array<IEntity> m_aQueryResults;
    protected vector m_vQueryCenter;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        // Initialize data structures
        m_CurrentSceneGraph = new SCR_SceneGraph();
        m_mPreviousPositions = new map<int, vector>();
        m_mPreviousTimestamps = new map<int, float>();
        m_mEntityIDs = new map<int, int>();
        m_aQueryResults = new array<IEntity>();

        m_iFrameCounter = 0;
        m_iNextEntityID = 1;
        m_fLastCaptureTime = 0;
        m_bOutputInitialized = false;

        // Try to find ego vehicle
        FindEgoVehicle(owner);

        Print("[SceneEnumerator] Component initialized on " + owner.GetName(), LogLevel.NORMAL);
        Print("[SceneEnumerator] Query radius: " + m_fQueryRadius.ToString() + "m, Max entities: " + m_iMaxEntities.ToString(), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Find the ego vehicle (either this entity or connected vehicle)
    protected void FindEgoVehicle(IEntity owner)
    {
        // Check if owner is a vehicle
        if (Vehicle.Cast(owner))
        {
            m_EgoVehicle = owner;
            Print("[SceneEnumerator] Ego vehicle: " + owner.GetName(), LogLevel.NORMAL);
            return;
        }

        // Check parent hierarchy for vehicle
        IEntity parent = owner.GetParent();
        while (parent)
        {
            if (Vehicle.Cast(parent))
            {
                m_EgoVehicle = parent;
                Print("[SceneEnumerator] Ego vehicle (parent): " + parent.GetName(), LogLevel.NORMAL);
                return;
            }
            parent = parent.GetParent();
        }

        // Use owner position as reference if no vehicle found
        m_EgoVehicle = owner;
        Print("[SceneEnumerator] Using owner as reference point: " + owner.GetName(), LogLevel.WARNING);
    }

    //------------------------------------------------------------------------
    // Set ego vehicle explicitly
    void SetEgoVehicle(IEntity vehicle)
    {
        m_EgoVehicle = vehicle;
        if (vehicle)
        {
            Print("[SceneEnumerator] Ego vehicle set to: " + vehicle.GetName(), LogLevel.NORMAL);
        }
    }

    //------------------------------------------------------------------------
    // Initialize CSV output file
    bool InitializeOutput(string sessionPath)
    {
        if (m_bOutputInitialized)
            return true;

        m_sOutputPath = sessionPath + "/scene_graph.csv";

        FileHandle file = FileIO.OpenFile(m_sOutputPath, FileMode.WRITE);
        if (!file)
        {
            Print("[SceneEnumerator] ERROR: Failed to create output file: " + m_sOutputPath, LogLevel.ERROR);
            return false;
        }

        // Write CSV header
        string header = "frame_id,timestamp_ms,entity_id,semantic_class,";
        header += "pos_x,pos_y,pos_z,";
        header += "fwd_x,fwd_y,fwd_z,";
        header += "bbox_min_x,bbox_min_y,bbox_min_z,";
        header += "bbox_max_x,bbox_max_y,bbox_max_z,";
        header += "vel_x,vel_y,vel_z,";
        header += "speed_ms,distance_to_ego";

        file.WriteLine(header);
        file.Close();

        m_bOutputInitialized = true;
        Print("[SceneEnumerator] Output initialized: " + m_sOutputPath, LogLevel.NORMAL);
        return true;
    }

    //------------------------------------------------------------------------
    // Main capture function - call this each frame or at desired frequency
    SCR_SceneGraph CaptureSceneGraph()
    {
        if (!m_EgoVehicle)
        {
            Print("[SceneEnumerator] ERROR: No ego vehicle set", LogLevel.ERROR);
            return null;
        }

        // Get world and current time
        World world = GetGame().GetWorld();
        if (!world)
            return null;

        float currentTime = world.GetWorldTime();

        // Clear previous results
        m_CurrentSceneGraph.Clear();
        m_aQueryResults.Clear();

        // Set frame metadata
        m_CurrentSceneGraph.m_iFrameID = m_iFrameCounter;
        m_CurrentSceneGraph.m_fTimestamp = currentTime;

        // Get ego vehicle position and state
        vector egoTransform[4];
        m_EgoVehicle.GetWorldTransform(egoTransform);
        vector egoPosition = egoTransform[3];
        m_vQueryCenter = egoPosition;

        // Capture ego vehicle data
        CaptureEgoVehicle(egoTransform, currentTime);

        // Query entities in sphere around ego
        // Note: QueryEntitiesBySphere uses callback functions
        // addEntity callback returns true to continue, false to stop
        // filterEntity callback returns true to include entity, false to skip
        world.QueryEntitiesBySphere(
            egoPosition,
            m_fQueryRadius,
            QueryAddEntityCallback,
            QueryFilterEntityCallback,
            EQueryEntitiesFlags.ALL
        );

        // Process query results
        ProcessQueryResults(egoPosition, currentTime);

        // Write to CSV if enabled
        if (m_bEnableCSVOutput && m_bOutputInitialized)
        {
            WriteSceneGraphToCSV();
        }

        // Update frame counter
        m_iFrameCounter++;
        m_fLastCaptureTime = currentTime;

        // Verbose logging
        if (m_bVerboseLogging && m_iFrameCounter % 50 == 0)
        {
            Print("[SceneEnumerator] Frame " + m_iFrameCounter.ToString() +
                  ": " + m_CurrentSceneGraph.GetTotalEntityCount().ToString() + " entities " +
                  "(V:" + m_CurrentSceneGraph.m_iVehicleCount.ToString() +
                  " P:" + m_CurrentSceneGraph.m_iPedestrianCount.ToString() +
                  " S:" + m_CurrentSceneGraph.m_iStaticCount.ToString() + ")", LogLevel.VERBOSE);
        }

        return m_CurrentSceneGraph;
    }

    //------------------------------------------------------------------------
    // Capture ego vehicle state
    protected void CaptureEgoVehicle(vector transform[4], float currentTime)
    {
        ref SCR_SceneEntityData egoData = new SCR_SceneEntityData();

        egoData.m_iEntityID = 0; // Ego always ID 0
        egoData.m_sEntityName = m_EgoVehicle.GetName();
        egoData.m_eSemanticClass = ESceneEntityClass.EGO_VEHICLE;

        // Position and orientation
        egoData.m_vPosition = transform[3];
        egoData.m_vRight = transform[0];
        egoData.m_vUp = transform[1];
        egoData.m_vForward = transform[2];

        // Get bounding box
        ExtractBoundingBox(m_EgoVehicle, egoData);

        // Get velocity
        ExtractVelocity(m_EgoVehicle, egoData, currentTime);

        egoData.m_fDistanceToEgo = 0;
        egoData.m_bIsTracked = true;

        m_CurrentSceneGraph.m_EgoVehicle = egoData;
    }

    //------------------------------------------------------------------------
    // Query filter callback - determines which entities to include
    // Returns true to include the entity in results, false to skip
    protected bool QueryFilterEntityCallback(IEntity entity)
    {
        if (!entity)
            return false;

        // Skip ego vehicle
        if (entity == m_EgoVehicle)
            return false;

        // Check entity limits
        if (m_aQueryResults.Count() >= m_iMaxEntities)
            return false;

        // Classify and filter by type
        ESceneEntityClass entityClass = ClassifyEntity(entity);

        switch (entityClass)
        {
            case ESceneEntityClass.OTHER_VEHICLE:
                return m_bIncludeVehicles;

            case ESceneEntityClass.PEDESTRIAN:
                return m_bIncludeCharacters;

            case ESceneEntityClass.STATIC_OBJECT:
            case ESceneEntityClass.BUILDING:
            case ESceneEntityClass.VEGETATION:
            case ESceneEntityClass.ROAD_INFRASTRUCTURE:
                if (!m_bIncludeStaticObjects)
                    return false;
                // Filter by minimum size
                return CheckMinimumSize(entity, m_fMinStaticSize);

            case ESceneEntityClass.DYNAMIC_PROP:
                return m_bIncludeDynamicProps;

            default:
                return false;
        }
    }

    //------------------------------------------------------------------------
    // Query add callback - adds entity to results
    // Returns true to continue query, false to stop
    protected bool QueryAddEntityCallback(IEntity entity)
    {
        if (entity && entity != m_EgoVehicle)
        {
            m_aQueryResults.Insert(entity);
        }
        return true; // Continue query
    }

    //------------------------------------------------------------------------
    // Process all query results and build scene graph
    protected void ProcessQueryResults(vector egoPosition, float currentTime)
    {
        foreach (IEntity entity : m_aQueryResults)
        {
            if (!entity)
                continue;

            // Create entity data
            ref SCR_SceneEntityData entityData = new SCR_SceneEntityData();

            // Get or assign entity ID
            int entityHash = entity.GetID();
            if (!m_mEntityIDs.Contains(entityHash))
            {
                m_mEntityIDs.Insert(entityHash, m_iNextEntityID);
                m_iNextEntityID++;
            }
            entityData.m_iEntityID = m_mEntityIDs.Get(entityHash);

            // Basic info
            entityData.m_sEntityName = entity.GetName();
            entityData.m_eSemanticClass = ClassifyEntity(entity);

            // Transform
            vector transform[4];
            entity.GetWorldTransform(transform);
            entityData.m_vPosition = transform[3];
            entityData.m_vRight = transform[0];
            entityData.m_vUp = transform[1];
            entityData.m_vForward = transform[2];

            // Bounding box
            ExtractBoundingBox(entity, entityData);

            // Velocity (if tracking enabled)
            if (m_bTrackVelocity)
            {
                ExtractVelocity(entity, entityData, currentTime);
            }

            // Distance to ego
            entityData.m_fDistanceToEgo = vector.Distance(egoPosition, entityData.m_vPosition);
            entityData.m_bIsTracked = true;

            // Add to scene graph
            m_CurrentSceneGraph.m_aEntities.Insert(entityData);

            // Update counters
            switch (entityData.m_eSemanticClass)
            {
                case ESceneEntityClass.OTHER_VEHICLE:
                    m_CurrentSceneGraph.m_iVehicleCount++;
                    break;
                case ESceneEntityClass.PEDESTRIAN:
                    m_CurrentSceneGraph.m_iPedestrianCount++;
                    break;
                case ESceneEntityClass.STATIC_OBJECT:
                case ESceneEntityClass.BUILDING:
                case ESceneEntityClass.VEGETATION:
                case ESceneEntityClass.ROAD_INFRASTRUCTURE:
                    m_CurrentSceneGraph.m_iStaticCount++;
                    break;
                case ESceneEntityClass.DYNAMIC_PROP:
                    m_CurrentSceneGraph.m_iDynamicCount++;
                    break;
            }
        }

        // Sort by distance to ego (closest first)
        SortEntitiesByDistance();
    }

    //------------------------------------------------------------------------
    // Classify entity by semantic type
    protected ESceneEntityClass ClassifyEntity(IEntity entity)
    {
        if (!entity)
            return ESceneEntityClass.UNKNOWN;

        // Check for vehicle
        Vehicle vehicle = Vehicle.Cast(entity);
        if (vehicle)
        {
            if (entity == m_EgoVehicle)
                return ESceneEntityClass.EGO_VEHICLE;
            return ESceneEntityClass.OTHER_VEHICLE;
        }

        // Check for vehicle controller component (alternative vehicle detection)
        VehicleControllerComponent vehicleController = VehicleControllerComponent.Cast(
            entity.FindComponent(VehicleControllerComponent)
        );
        if (vehicleController)
        {
            return ESceneEntityClass.OTHER_VEHICLE;
        }

        // Check for character
        SCR_ChimeraCharacter character = SCR_ChimeraCharacter.Cast(entity);
        if (character)
        {
            return ESceneEntityClass.PEDESTRIAN;
        }

        // Check for character controller component
        CharacterControllerComponent charController = CharacterControllerComponent.Cast(
            entity.FindComponent(CharacterControllerComponent)
        );
        if (charController)
        {
            return ESceneEntityClass.PEDESTRIAN;
        }

        // Check for physics to determine static vs dynamic
        Physics physics = entity.GetPhysics();
        if (physics)
        {
            // Dynamic objects have active physics simulation
            // Check if physics is active/dynamic
            vector velocity = physics.GetVelocity();
            float speed = velocity.Length();

            if (speed > 0.1) // Moving object
            {
                return ESceneEntityClass.DYNAMIC_PROP;
            }
        }

        // Check entity name for classification hints
        string name = entity.GetName();
        name.ToLower();

        // Building detection
        if (name.Contains("building") || name.Contains("house") ||
            name.Contains("structure") || name.Contains("barn") ||
            name.Contains("church") || name.Contains("factory"))
        {
            return ESceneEntityClass.BUILDING;
        }

        // Vegetation detection
        if (name.Contains("tree") || name.Contains("bush") ||
            name.Contains("forest") || name.Contains("grass") ||
            name.Contains("hedge") || name.Contains("plant"))
        {
            return ESceneEntityClass.VEGETATION;
        }

        // Road infrastructure
        if (name.Contains("sign") || name.Contains("barrier") ||
            name.Contains("fence") || name.Contains("pole") ||
            name.Contains("light") || name.Contains("bridge"))
        {
            return ESceneEntityClass.ROAD_INFRASTRUCTURE;
        }

        // Default to static object
        return ESceneEntityClass.STATIC_OBJECT;
    }

    //------------------------------------------------------------------------
    // Extract AABB bounding box from entity
    protected void ExtractBoundingBox(IEntity entity, SCR_SceneEntityData outData)
    {
        if (!entity)
            return;

        vector mins, maxs;

        // Try to get bounds directly from entity
        // Many entities support GetBounds()
        GenericEntity genericEntity = GenericEntity.Cast(entity);
        if (genericEntity)
        {
            genericEntity.GetBounds(mins, maxs);
        }
        else
        {
            // Fallback: estimate from visual object or use defaults
            mins = Vector(-1, 0, -1);
            maxs = Vector(1, 2, 1);
        }

        // Validate bounds
        if (mins == vector.Zero && maxs == vector.Zero)
        {
            // No valid bounds, use default vehicle-sized box
            mins = Vector(-2, 0, -4);
            maxs = Vector(2, 2, 4);
        }

        // Store bounds
        outData.m_vBoundsMin = mins;
        outData.m_vBoundsMax = maxs;

        // Calculate center and extents
        outData.m_vBoundsCenter = (mins + maxs) * 0.5;
        outData.m_vBoundsExtents = (maxs - mins) * 0.5;
    }

    //------------------------------------------------------------------------
    // Check if entity meets minimum size requirement
    protected bool CheckMinimumSize(IEntity entity, float minSize)
    {
        vector mins, maxs;

        GenericEntity genericEntity = GenericEntity.Cast(entity);
        if (genericEntity)
        {
            genericEntity.GetBounds(mins, maxs);
        }
        else
        {
            return true; // Can't determine size, include it
        }

        vector extents = maxs - mins;
        float maxExtent = Math.Max(Math.AbsFloat(extents[0]), Math.Max(Math.AbsFloat(extents[1]), Math.AbsFloat(extents[2])));

        return maxExtent >= minSize;
    }

    //------------------------------------------------------------------------
    // Extract velocity from physics or position delta
    protected void ExtractVelocity(IEntity entity, SCR_SceneEntityData outData, float currentTime)
    {
        if (!entity)
            return;

        int entityHash = entity.GetID();
        vector currentPos = outData.m_vPosition;

        // Try to get velocity from physics first
        Physics physics = entity.GetPhysics();
        if (physics)
        {
            outData.m_vVelocity = physics.GetVelocity();
            outData.m_fSpeed = outData.m_vVelocity.Length();

            // Get angular velocity if available
            // Note: Angular velocity method may vary by Enfusion version
            outData.m_vAngularVelocity = vector.Zero;
        }
        else
        {
            // Calculate velocity from position delta
            if (m_mPreviousPositions.Contains(entityHash) && m_mPreviousTimestamps.Contains(entityHash))
            {
                vector prevPos = m_mPreviousPositions.Get(entityHash);
                float prevTime = m_mPreviousTimestamps.Get(entityHash);

                float deltaTime = (currentTime - prevTime) / 1000.0; // Convert ms to seconds

                if (deltaTime > 0.001) // Avoid division by zero
                {
                    outData.m_vVelocity = (currentPos - prevPos) / deltaTime;
                    outData.m_fSpeed = outData.m_vVelocity.Length();
                }
            }
        }

        // Update tracking maps
        m_mPreviousPositions.Set(entityHash, currentPos);
        m_mPreviousTimestamps.Set(entityHash, currentTime);
    }

    //------------------------------------------------------------------------
    // Sort entities by distance to ego (bubble sort for simplicity)
    protected void SortEntitiesByDistance()
    {
        int n = m_CurrentSceneGraph.m_aEntities.Count();

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = 0; j < n - i - 1; j++)
            {
                if (m_CurrentSceneGraph.m_aEntities[j].m_fDistanceToEgo >
                    m_CurrentSceneGraph.m_aEntities[j + 1].m_fDistanceToEgo)
                {
                    // Swap
                    ref SCR_SceneEntityData temp = m_CurrentSceneGraph.m_aEntities[j];
                    m_CurrentSceneGraph.m_aEntities[j] = m_CurrentSceneGraph.m_aEntities[j + 1];
                    m_CurrentSceneGraph.m_aEntities[j + 1] = temp;
                }
            }
        }
    }

    //------------------------------------------------------------------------
    // Write current scene graph to CSV
    protected void WriteSceneGraphToCSV()
    {
        FileHandle file = FileIO.OpenFile(m_sOutputPath, FileMode.APPEND);
        if (!file)
            return;

        int frameID = m_CurrentSceneGraph.m_iFrameID;
        float timestamp = m_CurrentSceneGraph.m_fTimestamp;

        // Write ego vehicle
        if (m_CurrentSceneGraph.m_EgoVehicle)
        {
            string row = frameID.ToString() + "," + timestamp.ToString(12, 1) + ",";
            row += m_CurrentSceneGraph.m_EgoVehicle.ToCSVRow();
            file.WriteLine(row);
        }

        // Write all other entities
        foreach (SCR_SceneEntityData entity : m_CurrentSceneGraph.m_aEntities)
        {
            string row = frameID.ToString() + "," + timestamp.ToString(12, 1) + ",";
            row += entity.ToCSVRow();
            file.WriteLine(row);
        }

        file.Close();
    }

    //------------------------------------------------------------------------
    // Get current scene graph (for external access)
    SCR_SceneGraph GetCurrentSceneGraph()
    {
        return m_CurrentSceneGraph;
    }

    //------------------------------------------------------------------------
    // Get entity count by class
    int GetEntityCountByClass(ESceneEntityClass eClass)
    {
        int count = 0;
        foreach (SCR_SceneEntityData entity : m_CurrentSceneGraph.m_aEntities)
        {
            if (entity.m_eSemanticClass == eClass)
                count++;
        }
        return count;
    }

    //------------------------------------------------------------------------
    // Get closest entity of a specific class
    SCR_SceneEntityData GetClosestEntityOfClass(ESceneEntityClass eClass)
    {
        foreach (SCR_SceneEntityData entity : m_CurrentSceneGraph.m_aEntities)
        {
            if (entity.m_eSemanticClass == eClass)
                return entity; // Already sorted by distance
        }
        return null;
    }

    //------------------------------------------------------------------------
    // Get all entities within a specific distance
    array<ref SCR_SceneEntityData> GetEntitiesWithinDistance(float distance)
    {
        array<ref SCR_SceneEntityData> result = new array<ref SCR_SceneEntityData>();

        foreach (SCR_SceneEntityData entity : m_CurrentSceneGraph.m_aEntities)
        {
            if (entity.m_fDistanceToEgo <= distance)
                result.Insert(entity);
        }

        return result;
    }

    //------------------------------------------------------------------------
    // Get frame count
    int GetFrameCount()
    {
        return m_iFrameCounter;
    }

    //------------------------------------------------------------------------
    // Clear velocity tracking history (call when teleporting or scene changes)
    void ClearVelocityHistory()
    {
        m_mPreviousPositions.Clear();
        m_mPreviousTimestamps.Clear();
        Print("[SceneEnumerator] Velocity history cleared", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Debug visualization - draw bounding boxes
    void DebugDrawSceneGraph(float duration = 0.0)
    {
        if (!m_CurrentSceneGraph)
            return;

        // Draw ego vehicle box (green)
        if (m_CurrentSceneGraph.m_EgoVehicle)
        {
            DebugDrawEntityBox(m_CurrentSceneGraph.m_EgoVehicle, 0xFF00FF00, duration);
        }

        // Draw other entities
        foreach (SCR_SceneEntityData entity : m_CurrentSceneGraph.m_aEntities)
        {
            int color;
            switch (entity.m_eSemanticClass)
            {
                case ESceneEntityClass.OTHER_VEHICLE:
                    color = 0xFFFF0000; // Red
                    break;
                case ESceneEntityClass.PEDESTRIAN:
                    color = 0xFFFFFF00; // Yellow
                    break;
                case ESceneEntityClass.DYNAMIC_PROP:
                    color = 0xFFFF8800; // Orange
                    break;
                default:
                    color = 0xFF888888; // Gray
                    break;
            }

            DebugDrawEntityBox(entity, color, duration);
        }
    }

    //------------------------------------------------------------------------
    // Draw a single entity's bounding box
    protected void DebugDrawEntityBox(SCR_SceneEntityData entity, int color, float duration)
    {
        // Calculate world-space corners of the AABB
        vector pos = entity.m_vPosition;
        vector mins = entity.m_vBoundsMin;
        vector maxs = entity.m_vBoundsMax;

        // Transform local bounds to world space
        vector right = entity.m_vRight;
        vector up = entity.m_vUp;
        vector forward = entity.m_vForward;

        // Calculate 8 corners of the OBB in world space
        // For simplicity, draw AABB at world position
        vector worldMins = pos + mins;
        vector worldMaxs = pos + maxs;

        // Draw wireframe box using Shape API
        Shape box = Shape.Create(
            ShapeType.BBOX,
            color,
            ShapeFlags.VISIBLE | ShapeFlags.WIREFRAME | ShapeFlags.NOOUTLINE,
            worldMins,
            worldMaxs
        );

        // Note: Shape will persist until deleted or scene cleared
    }

    //------------------------------------------------------------------------
    // Cleanup
    override void OnDelete(IEntity owner)
    {
        m_CurrentSceneGraph = null;
        m_mPreviousPositions = null;
        m_mPreviousTimestamps = null;
        m_mEntityIDs = null;
        m_aQueryResults = null;

        super.OnDelete(owner);
    }
}

// -----------------------------------------------------------------------------
// Scene Enumerator Manager (optional singleton for global access)
// -----------------------------------------------------------------------------
class SCR_SceneEnumeratorManager
{
    protected static SCR_SceneEnumerator s_Instance;

    //------------------------------------------------------------------------
    static void SetInstance(SCR_SceneEnumerator instance)
    {
        s_Instance = instance;
    }

    //------------------------------------------------------------------------
    static SCR_SceneEnumerator GetInstance()
    {
        return s_Instance;
    }

    //------------------------------------------------------------------------
    static SCR_SceneGraph CaptureScene()
    {
        if (s_Instance)
            return s_Instance.CaptureSceneGraph();
        return null;
    }
}
