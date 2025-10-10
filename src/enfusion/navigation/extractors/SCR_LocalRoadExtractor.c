// Local Road Extractor - Extracts and visualizes roads around the component position
[ComponentEditorProps(category: "GameScripted/AI/Navigation", description: "Local Road Extractor and Visualizer")]
class SCR_LocalRoadExtractorClass: ScriptComponentClass
{
}

class SCR_LocalRoadExtractor: ScriptComponent
{
    [Attribute("500", UIWidgets.Slider, "Search radius around component (in meters)", "50 2000 50")]
    protected float m_fSearchRadius;
    
    [Attribute("1", UIWidgets.CheckBox, "Visualize roads in debug mode")]
    protected bool m_bDebugVisualize;
    
    [Attribute("0", UIWidgets.CheckBox, "Auto-update visualization on component movement")]
    protected bool m_bAutoUpdate;
    
    [Attribute("2.0", UIWidgets.Slider, "Size of visualization spheres", "0.1 10.0 0.1")]
    protected float m_fVisualizationSize;
    
    [Attribute("1", UIWidgets.CheckBox, "Clear previous visualization when updating")]
    protected bool m_bClearOnUpdate;
    
    [Attribute("3.0", UIWidgets.Slider, "Auto-update interval (seconds)", "0.5 10.0 0.5")]
    protected float m_fUpdateInterval;
    
    [Attribute("75", UIWidgets.Slider, "Maximum roads to visualize", "10 200 5")]
    protected int m_iMaxRoadsToVisualize;
    
    [Attribute("10", UIWidgets.Slider, "Points to visualize per road", "1 50 1")]
    protected int m_iPointsPerRoad;

    protected SCR_AIWorld m_AIWorld;
    protected RoadNetworkManager m_RoadNetworkManager;
    
    // Store for visualization and access
    protected ref array<vector> m_VisualizationPoints = {};
    protected ref array<Shape> m_DebugShapes = {};
    protected ref array<ref array<vector>> m_RoadPaths = {};
    
    // Last position for auto-update
    protected vector m_LastPosition;
    protected float m_UpdateTimer;
    protected bool m_bFrameEventSet;
    
    //------------------------------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);
        
        // Initialize arrays
        m_VisualizationPoints = new array<vector>();
        m_DebugShapes = new array<Shape>();
        m_RoadPaths = new array<ref array<vector>>();
        
        Print("SCR_LocalRoadExtractor: Initializing...", LogLevel.NORMAL);
        
        // Initialize our AI world and road network manager
        InitializeRoadNetworkManager();
        
        // Store initial position
        m_LastPosition = owner.GetOrigin();
        
        // Initial extraction
        ExtractRoadData();
        
        // Setup update if needed
        if (m_bAutoUpdate)
        {
            // Enable frame event
            SetEventMask(owner, EntityEvent.FRAME);
            m_bFrameEventSet = true;
            m_UpdateTimer = 0;
        }
    }
    
    //------------------------------------------------------------------------------------------------
    bool InitializeRoadNetworkManager()
    {
        // Get the AI world instance
        m_AIWorld = SCR_AIWorld.Cast(GetGame().GetAIWorld());
        
        if (!m_AIWorld)
        {
            Print("SCR_LocalRoadExtractor: ERROR - AIWorld not found!", LogLevel.ERROR);
            return false;
        }
        
        // Get the road network manager
        m_RoadNetworkManager = m_AIWorld.GetRoadNetworkManager();
        if (!m_RoadNetworkManager)
        {
            Print("SCR_LocalRoadExtractor: ERROR - RoadNetworkManager not available.", LogLevel.ERROR);
            return false;
        }
        
        Print("SCR_LocalRoadExtractor: Successfully connected to road network manager.");
        return true;
    }
    
    //------------------------------------------------------------------------------------------------
    // Extract road data around the component's position
    void ExtractRoadData()
    {
        IEntity owner = GetOwner();
        if (!owner)
            return;
            
        // Get the current position as the center of search
        vector centerPos = owner.GetOrigin();
        
        // Store for change detection
        m_LastPosition = centerPos;
        
        // Clear previous data if needed
        if (m_bClearOnUpdate)
        {
            ClearShapes();
            m_VisualizationPoints.Clear();
            m_RoadPaths.Clear();
        }
        else
        {
            // If not clearing fully, at least clear shapes
            ClearShapes();
        }
        
        // Get roads around position
        array<BaseRoad> nearbyRoads = {};
        GetRoadsAroundPosition(centerPos, m_fSearchRadius, nearbyRoads);
        
        int roadCount = nearbyRoads.Count();
        if (roadCount <= 0)
        {
            Print("SCR_LocalRoadExtractor: No roads found within " + m_fSearchRadius + "m radius.", LogLevel.WARNING);
            return;
        }
        
        // Limit roads if needed
        int roadsToProcess = roadCount;
        if (roadsToProcess > m_iMaxRoadsToVisualize)
        {
            roadsToProcess = m_iMaxRoadsToVisualize;
            Print("SCR_LocalRoadExtractor: Limiting from " + roadCount + " to " + roadsToProcess + " roads");
        }
        
        // Process each road
        for (int roadIndex = 0; roadIndex < roadsToProcess; roadIndex++)
        {
            BaseRoad road = nearbyRoads[roadIndex];
            
            // Temporary array for points
            array<vector> roadPoints = {};
            road.GetPoints(roadPoints);
            
            if (roadPoints.IsEmpty())
                continue;
            
            // Store this road's points
            ref array<vector> storedRoadPath = new array<vector>();
            storedRoadPath.Copy(roadPoints);
            m_RoadPaths.Insert(storedRoadPath);
            
            // Calculate visualization points
            int totalPoints = roadPoints.Count();
            int step = Math.Max(1, totalPoints / m_iPointsPerRoad);
            
            // Add visualization points
            for (int i = 0; i < totalPoints; i += step)
            {
                if (i < totalPoints) // Safety check
                {
                    m_VisualizationPoints.Insert(roadPoints[i]);
                }
            }
            
            // Print basic info for debugging
            vector start = roadPoints[0];
            vector end = roadPoints[roadPoints.Count() - 1];
            PrintFormat("Road[%1]: %2 points (start=%3, end=%4)", 
                roadIndex, roadPoints.Count(), start, end);
        }
        
        PrintFormat("SCR_LocalRoadExtractor: Found %1 roads, %2 visualization points around %3", 
            roadsToProcess, m_VisualizationPoints.Count(), centerPos);
            
        // Create visualization if enabled
        if (m_bDebugVisualize)
        {
            CreateVisualization();
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Get roads around a specific position
    void GetRoadsAroundPosition(vector position, float radius, out array<BaseRoad> outRoads)
    {
        if (!m_RoadNetworkManager)
        {
            if (!InitializeRoadNetworkManager())
                return;
        }
        
        // Create a bounding box around the position
        vector min = position - Vector(radius, radius, radius);
        vector max = position + Vector(radius, radius, radius);
        
        // Get roads in this area
        m_RoadNetworkManager.GetRoadsInAABB(min, max, outRoads);
    }
    
    //------------------------------------------------------------------------------------------------
    // Create visual debug shapes for roads
    void CreateVisualization()
    {
        // Clear previous shapes
        ClearShapes();
        
        int pointCount = m_VisualizationPoints.Count();
        Print("SCR_LocalRoadExtractor: Creating visualization for " + pointCount + " road points");
        
        if (pointCount == 0)
        {
            Print("SCR_LocalRoadExtractor: WARNING - No points to visualize!", LogLevel.WARNING);
            return;
        }
        
        // Choose color based on number of roads (more roads = different color)
        int roadCount = m_RoadPaths.Count();
        int color = COLOR_GREEN;  // Default green
        
        if (roadCount > 30)
            color = COLOR_RED;  // Red
        else if (roadCount > 10)
            color = COLOR_YELLOW;  // Yellow
        
        // Draw points
        for (int i = 0; i < pointCount; i++)
        {
            if (!m_VisualizationPoints[i])
                continue;
                
            // Create a sphere at each road point
            Shape sphere = Shape.CreateSphere(
                color, 
                ShapeFlags.VISIBLE | ShapeFlags.NOZBUFFER, 
                m_VisualizationPoints[i], 
                m_fVisualizationSize
            );
            
            // Store for later cleanup
            if (sphere)
            {
                m_DebugShapes.Insert(sphere);
            }
        }
        
        Print("SCR_LocalRoadExtractor: Created " + m_DebugShapes.Count() + " visualization shapes");
    }
    
    //------------------------------------------------------------------------------------------------
    // Clear just the shapes, not the points
    void ClearShapes()
    {
        foreach (Shape shape : m_DebugShapes)
        {
            if (shape)
            {
                delete shape;
            }
        }
        
        m_DebugShapes.Clear();
    }
    
    //------------------------------------------------------------------------------------------------
    // Frame update for auto-update feature
    override void EOnFrame(IEntity owner, float timeSlice)
    {
        super.EOnFrame(owner, timeSlice);
        
        if (!m_bAutoUpdate || !owner)
            return;
            
        // Update timer
        m_UpdateTimer += timeSlice;
        
        // Check if it's time to update
        if (m_UpdateTimer >= m_fUpdateInterval)
        {
            m_UpdateTimer = 0;
            
            // Get current position
            vector currentPos = owner.GetOrigin();
            
            // Check if we've moved enough to update
            float moveDistance = vector.Distance(currentPos, m_LastPosition);
            if (moveDistance > 5.0) // Only update if moved more than 5 meters
            {
                ExtractRoadData();
            }
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Get the closest road point to a position
    bool GetClosestRoadPoint(vector position, out vector closestPoint)
    {
        float minDistance = float.MAX;
        bool found = false;
        
        // Check each stored road
        foreach (array<vector> roadPath : m_RoadPaths)
        {
            // Find closest point on this road
            foreach (vector point : roadPath)
            {
                float distance = vector.Distance(position, point);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closestPoint = point;
                    found = true;
                }
            }
        }
        
        return found;
    }
    
    //------------------------------------------------------------------------------------------------
    // Get the nearest road segment (as an array of points)
    array<vector> GetNearestRoadSegment(vector position, float maxDistance = 50)
    {
        if (m_RoadPaths.IsEmpty())
            return null;
            
        float minDistance = float.MAX;
        int bestRoadIndex = -1;
        
        // Find the closest road
        for (int i = 0; i < m_RoadPaths.Count(); i++)
        {
            array<vector> roadPath = m_RoadPaths[i];
            if (roadPath.IsEmpty())
                continue;
                
            // Check the first point as an approximation
            float distance = vector.Distance(position, roadPath[0]);
            
            // Also check the last point
            float distanceEnd = vector.Distance(position, roadPath[roadPath.Count() - 1]);
            if (distanceEnd < distance)
                distance = distanceEnd;
                
            // Check a middle point too for better approximation
            if (roadPath.Count() > 2)
            {
                float distMid = vector.Distance(position, roadPath[roadPath.Count() / 2]);
                if (distMid < distance)
                    distance = distMid;
            }
            
            if (distance < minDistance)
            {
                minDistance = distance;
                bestRoadIndex = i;
            }
        }
        
        // Return the best road if within max distance
        if (bestRoadIndex >= 0 && minDistance <= maxDistance)
        {
            return m_RoadPaths[bestRoadIndex];
        }
        
        return null;
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API - Manual trigger for road extraction
    void ManualExtract()
    {
        ExtractRoadData();
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API - Toggle visualization
    void ToggleVisualization(bool enable)
    {
        m_bDebugVisualize = enable;
        
        if (enable)
        {
            CreateVisualization();
        }
        else
        {
            ClearShapes();
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API - Toggle auto-update
    void SetAutoUpdate(bool enable, float interval = -1)
    {
        m_bAutoUpdate = enable;
        
        if (interval > 0)
        {
            m_fUpdateInterval = interval;
        }
        
        // Update event mask if needed
        IEntity owner = GetOwner();
        if (!owner)
            return;
            
        if (enable && !m_bFrameEventSet)
        {
            SetEventMask(owner, EntityEvent.FRAME);
            m_bFrameEventSet = true;
            m_UpdateTimer = 0;
        }
        else if (!enable && m_bFrameEventSet)
        {
            ClearEventMask(owner, EntityEvent.FRAME);
            m_bFrameEventSet = false;
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API - Get all stored road paths
    array<ref array<vector>> GetAllRoadPaths()
    {
        return m_RoadPaths;
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API - Get visualization points
    array<vector> GetVisualizationPoints()
    {
        return m_VisualizationPoints;
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API - Get a simple path along a road from origin to a target position
    bool GetRoadPathTo(vector originPos, vector targetPos, out array<vector> pathPoints, float maxSearchRadius = 300)
    {
        // Get roads around both positions
        array<BaseRoad> originRoads = {};
        array<BaseRoad> targetRoads = {};
        
        GetRoadsAroundPosition(originPos, maxSearchRadius, originRoads);
        GetRoadsAroundPosition(targetPos, maxSearchRadius, targetRoads);
        
        if (originRoads.IsEmpty() || targetRoads.IsEmpty())
            return false;
            
        // Find closest roads to origin and target
        vector closestOriginPoint, closestTargetPoint;
        BaseRoad originRoad = null;
        BaseRoad targetRoad = null;
        
        float minOriginDist = float.MAX;
        float minTargetDist = float.MAX;
        
        // Find closest road to origin
        foreach (BaseRoad road : originRoads)
        {
            array<vector> points = {};
            road.GetPoints(points);
            
            foreach (vector point : points)
            {
                float dist = vector.Distance(originPos, point);
                if (dist < minOriginDist)
                {
                    minOriginDist = dist;
                    closestOriginPoint = point;
                    originRoad = road;
                }
            }
        }
        
        // Find closest road to target
        foreach (BaseRoad road : targetRoads)
        {
            array<vector> points = {};
            road.GetPoints(points);
            
            foreach (vector point : points)
            {
                float dist = vector.Distance(targetPos, point);
                if (dist < minTargetDist)
                {
                    minTargetDist = dist;
                    closestTargetPoint = point;
                    targetRoad = road;
                }
            }
        }
        
        // Check if the same road segment connects both points (simple case)
        if (originRoad == targetRoad)
        {
            // Get all points of this road
            array<vector> roadPoints = {};
            originRoad.GetPoints(roadPoints);
            
            // Find indices of closest points
            int originIndex = -1;
            int targetIndex = -1;
            
            for (int i = 0; i < roadPoints.Count(); i++)
            {
                if (roadPoints[i] == closestOriginPoint)
                    originIndex = i;
                    
                if (roadPoints[i] == closestTargetPoint)
                    targetIndex = i;
            }
            
            // Create path between the two points
            if (originIndex >= 0 && targetIndex >= 0)
            {
                pathPoints = new array<vector>();
                
                int start = Math.Min(originIndex, targetIndex);
                int end = Math.Max(originIndex, targetIndex);
                
                for (int i = start; i <= end; i++)
                {
                    pathPoints.Insert(roadPoints[i]);
                }
                
                return true;
            }
        }
        
        // TODO: For more complex cases, implement proper pathfinding between roads
        // This would require traversing the road network graph, which is beyond the scope
        // of this simple method. For now, return false for different roads.
        
        return false;
    }
}