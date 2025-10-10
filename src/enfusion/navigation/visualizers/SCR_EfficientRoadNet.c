// Enhanced Road Visualizer for AI Driving Debugging
// This component provides advanced visualization features for roads and AI path generation
[ComponentEditorProps(category: "GameScripted/AI/Navigation", description: "Enhanced Road Visualizer for AI Driving Debug")]
class SCR_EnhancedRoadVisualizerClass: ScriptComponentClass
{
}

class SCR_EnhancedRoadVisualizer: ScriptComponent
{
    [Attribute("500", UIWidgets.Slider, "Search radius around component (in meters)", "50 2000 50")]
    protected float m_fSearchRadius;
    
    [Attribute("1", UIWidgets.CheckBox, "Auto-update visualization on component movement")]
    protected bool m_bAutoUpdate;
    
    [Attribute("3.0", UIWidgets.Slider, "Auto-update interval (seconds)", "0.5 10.0 0.5")]
    protected float m_fUpdateInterval;
    
    [Attribute("1", UIWidgets.CheckBox, "Draw road surface centerlines")]
    protected bool m_bDrawRoadLines;
    
    [Attribute("1", UIWidgets.CheckBox, "Draw road nodes/points")]
    protected bool m_bDrawRoadNodes;
    
    [Attribute("1", UIWidgets.CheckBox, "Draw road junctions/intersections")]
    protected bool m_bDrawJunctions;
    
    [Attribute("0", UIWidgets.CheckBox, "Draw all potential path directions")]
    protected bool m_bDrawPathOptions;
    
    [Attribute("1", UIWidgets.CheckBox, "Draw path prediction for this entity")]
    protected bool m_bDrawPathPrediction;
    
    [Attribute("1.0", UIWidgets.Slider, "Road line width", "0.1 5.0 0.1")]
    protected float m_fRoadLineWidth;
    
    [Attribute("1.0", UIWidgets.Slider, "Node marker size", "0.1 5.0 0.1")]
    protected float m_fNodeSize;
    
    [Attribute("2.0", UIWidgets.Slider, "Junction marker size", "0.5 10.0 0.5")]
    protected float m_fJunctionSize;
    
    [Attribute("100", UIWidgets.Slider, "Maximum roads to process", "10 500 10")]
    protected int m_iMaxRoadsToProcess;
    
    [Attribute("5", UIWidgets.Slider, "Path prediction distance (in segments)", "1 20 1")]
    protected int m_iPathPredictionSegments;
    
    [Attribute("10", UIWidgets.Slider, "Distance between road surface visualizers", "1 50 1")]
    protected float m_fRoadLineDensity;
    
    [Attribute("40", UIWidgets.Slider, "Height of road prediction lines above ground", "1 100 1")]
    protected float m_fPathHeight;

    // Road highlighting color options
    [Attribute("", UIWidgets.ColorPicker, "Road segment color")]
    protected ref Color m_RoadSegmentColor;
    
    [Attribute("", UIWidgets.ColorPicker, "Node point color")]
    protected ref Color m_NodeColor;
    
    [Attribute("", UIWidgets.ColorPicker, "Junction color")]
    protected ref Color m_JunctionColor;
    
    [Attribute("", UIWidgets.ColorPicker, "Path prediction color")]
    protected ref Color m_PathColor;
    
    protected SCR_AIWorld m_AIWorld;
    protected RoadNetworkManager m_RoadNetworkManager;
    
    // Visualization storage
    protected ref array<Shape> m_DebugShapes = {};
    protected ref array<ref array<vector>> m_RoadPaths = {};
    protected ref array<vector> m_JunctionPoints = {};
    
    // Path finding and prediction
    protected ref array<vector> m_PredictedPath = {};
    protected vector m_LastKnownDirection;
    
    // Tracking
    protected vector m_LastPosition;
    protected float m_UpdateTimer;
    protected bool m_bFrameEventSet;
    
    // Connected vehicle for AI path prediction
    protected Vehicle m_ConnectedVehicle;
    
    //------------------------------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);
        
        // Set default colors if not specified
        if (!m_RoadSegmentColor)
            m_RoadSegmentColor = Color.FromARGB(0xFF00FF00); // Green
            
        if (!m_NodeColor)
            m_NodeColor = Color.FromARGB(0xFFFFFFFF); // White
            
        if (!m_JunctionColor)
            m_JunctionColor = Color.FromARGB(0xFFFF0000); // Red
            
        if (!m_PathColor)
            m_PathColor = Color.FromARGB(0xFF00FFFF); // Cyan
        
        // Initialize our arrays
        m_DebugShapes = new array<Shape>();
        m_RoadPaths = new array<ref array<vector>>();
        m_JunctionPoints = new array<vector>();
        m_PredictedPath = new array<vector>();
        
        Print("SCR_EnhancedRoadVisualizer: Initializing...", LogLevel.NORMAL);
        
        // Try to find and connect to a vehicle component on this entity
        if (Vehicle.Cast(owner))
        {
            m_ConnectedVehicle = Vehicle.Cast(owner);
            Print("SCR_EnhancedRoadVisualizer: Connected to vehicle " + owner.GetName());
        }
        else
        {
            // Try to find a vehicle component on a parent
            IEntity parent = owner.GetParent();
            while (parent)
            {
                if (Vehicle.Cast(parent))
                {
                    m_ConnectedVehicle = Vehicle.Cast(parent);
                    Print("SCR_EnhancedRoadVisualizer: Connected to parent vehicle " + parent.GetName());
                    break;
                }
                parent = parent.GetParent();
            }
        }
        
        // Initialize our AI world and road network manager
        InitializeRoadNetworkManager();
        
        // Store initial position
        m_LastPosition = owner.GetOrigin();
        
        // Initial extraction and visualization
        ExtractAndVisualize();
        
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
            Print("SCR_EnhancedRoadVisualizer: ERROR - AIWorld not found!", LogLevel.ERROR);
            return false;
        }
        
        // Get the road network manager
        m_RoadNetworkManager = m_AIWorld.GetRoadNetworkManager();
        if (!m_RoadNetworkManager)
        {
            Print("SCR_EnhancedRoadVisualizer: ERROR - RoadNetworkManager not available.", LogLevel.ERROR);
            return false;
        }
        
        Print("SCR_EnhancedRoadVisualizer: Successfully connected to road network manager.");
        return true;
    }
    
    //------------------------------------------------------------------------------------------------
    // Main method to extract and visualize road data
    void ExtractAndVisualize()
    {
        IEntity owner = GetOwner();
        if (!owner)
            return;
            
        // Get the current position as the center of search
        vector centerPos = owner.GetOrigin();
        
        // Store for change detection
        m_LastPosition = centerPos;
        
        // Clear previous shapes
        ClearShapes();
        
        // Clear data
        m_RoadPaths.Clear();
        m_JunctionPoints.Clear();
        m_PredictedPath.Clear();
        
        // Get roads around position
        array<BaseRoad> nearbyRoads = {};
        GetRoadsAroundPosition(centerPos, m_fSearchRadius, nearbyRoads);
        
        int roadCount = nearbyRoads.Count();
        if (roadCount <= 0)
        {
            Print("SCR_EnhancedRoadVisualizer: No roads found within " + m_fSearchRadius + "m radius.", LogLevel.WARNING);
            return;
        }
        
        // Limit roads if needed
        int roadsToProcess = roadCount;
        if (roadsToProcess > m_iMaxRoadsToProcess)
        {
            roadsToProcess = m_iMaxRoadsToProcess;
            Print("SCR_EnhancedRoadVisualizer: Limiting from " + roadCount + " to " + roadsToProcess + " roads");
        }
        
        // Build road network representation
        BuildRoadNetwork(nearbyRoads, roadsToProcess);
        
        // Find and store junctions
        FindJunctions();
        
        // Generate path prediction if connected to a vehicle
        if (m_bDrawPathPrediction && m_ConnectedVehicle)
        {
            GeneratePathPrediction();
        }
        
        // Create visual elements
        CreateVisualizations();
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
    // Build a network representation of the roads for better analysis
    void BuildRoadNetwork(array<BaseRoad> roads, int maxRoads)
    {
        // Process each road
        for (int roadIndex = 0; roadIndex < Math.Min(maxRoads, roads.Count()); roadIndex++)
        {
            BaseRoad road = roads[roadIndex];
            
            // Get points for this road
            array<vector> roadPoints = {};
            road.GetPoints(roadPoints);
            
            if (roadPoints.IsEmpty())
                continue;
            
            // Store the points
            ref array<vector> storedRoadPath = new array<vector>();
            storedRoadPath.Copy(roadPoints);
            m_RoadPaths.Insert(storedRoadPath);
            
            // Log basic info
            vector start = roadPoints[0];
            vector end = roadPoints[roadPoints.Count() - 1];
            PrintFormat("Road[%1]: %2 points (start=%3, end=%4)", 
                roadIndex, roadPoints.Count(), start, end);
        }
        
        Print("SCR_EnhancedRoadVisualizer: Processed " + m_RoadPaths.Count() + " roads");
    }
    
    //------------------------------------------------------------------------------------------------
    // Find road junctions - places where roads intersect
    void FindJunctions()
    {
        // A junction is a point where multiple roads connect
        // Simple approach: look for start/end points that are shared by multiple roads
        
        // Track potential junction points
        ref map<string, int> junctionCounts = new map<string, int>();
        
        // Check start and end points of all roads
        foreach (array<vector> road : m_RoadPaths)
        {
            if (road.Count() < 2)
                continue;
                
            // Check first and last point
            vector start = road[0];
            vector end = road[road.Count() - 1];
            
            // Create keys with reduced precision to handle slight differences
            string startKey = string.Format("%.1f,%.1f,%.1f", start[0], start[1], start[2]);
            string endKey = string.Format("%.1f,%.1f,%.1f", end[0], end[1], end[2]);
            
            // Increment counts
            if (!junctionCounts.Contains(startKey))
                junctionCounts.Insert(startKey, 1);
            else
                junctionCounts[startKey] = junctionCounts[startKey] + 1;
                
            if (!junctionCounts.Contains(endKey))
                junctionCounts.Insert(endKey, 1);
            else
                junctionCounts[endKey] = junctionCounts[endKey] + 1;
        }
        
        // Add points that are referenced by multiple roads (junctions)
        foreach (string key, int count : junctionCounts)
        {
            if (count >= 2) // Points shared by at least 2 roads
            {
                // Parse back to vector
                array<string> parts = {};
                key.Split(",", parts);
                if (parts.Count() == 3)
                {
                    vector junctionPos = Vector(parts[0].ToFloat(), parts[1].ToFloat(), parts[2].ToFloat());
                    m_JunctionPoints.Insert(junctionPos);
                }
            }
        }
        
        Print("SCR_EnhancedRoadVisualizer: Found " + m_JunctionPoints.Count() + " road junctions");
    }
    
    //------------------------------------------------------------------------------------------------
    // Generate a predicted path for the connected vehicle
    void GeneratePathPrediction()
    {
        if (!m_ConnectedVehicle)
            return;
            
        // Get current position and rotation
        vector pos = m_ConnectedVehicle.GetOrigin();
        vector mat[4];
        m_ConnectedVehicle.GetTransform(mat);
        
        // Get vehicle's forward direction 
        vector dir = mat[2];
        
        // Store this for future reference
        if (vector.Dot(dir, vector.Forward) > 0.7)
            m_LastKnownDirection = dir;
        
        // Find closest road
        array<vector> closestRoad = GetClosestRoadSegment(pos);
        if (!closestRoad || closestRoad.Count() < 2)
        {
            // No road found, try to use last known direction
            if (m_LastKnownDirection != vector.Zero)
            {
                // Create a simple straight line prediction
                m_PredictedPath.Insert(pos);
                
                for (int i = 1; i <= m_iPathPredictionSegments; i++)
                {
                    vector nextPos = pos + m_LastKnownDirection * (50.0 * i);
                    // Set height above ground
                    nextPos[1] = GetGame().GetWorld().GetSurfaceY(nextPos[0], nextPos[2]) + m_fPathHeight;
                    m_PredictedPath.Insert(nextPos);
                }
            }
            return;
        }
        
        // Find closest point on the road
        int closestIndex = 0;
        float minDist = float.MAX;
        
        for (int i = 0; i < closestRoad.Count(); i++)
        {
            float dist = vector.Distance(pos, closestRoad[i]);
            if (dist < minDist)
            {
                minDist = dist;
                closestIndex = i;
            }
        }
        
        // Determine which direction to follow based on vehicle direction
        bool forwardDirection = true;
        
        if (closestIndex < closestRoad.Count() - 1)
        {
            vector roadDir = closestRoad[closestIndex + 1] - closestRoad[closestIndex];
            roadDir.Normalize();
            
            // Compare with vehicle direction
            float dotProduct = vector.Dot(dir, roadDir);
            forwardDirection = dotProduct >= 0;
        }
        
        // Generate path prediction
        m_PredictedPath.Clear();
        m_PredictedPath.Insert(pos); // Start with current position
        
        // Add road points in the appropriate direction
        int segmentsAdded = 0;
        
        if (forwardDirection)
        {
            // Forward direction (towards end of road)
            for (int i = closestIndex; i < closestRoad.Count() && segmentsAdded < m_iPathPredictionSegments; i++)
            {
                vector pathPoint = closestRoad[i];
                // Raise height for visibility
                pathPoint[1] += m_fPathHeight;
                m_PredictedPath.Insert(pathPoint);
                segmentsAdded++;
            }
            
            // If we need more segments, look for connected roads
            if (segmentsAdded < m_iPathPredictionSegments && closestRoad.Count() > 0)
            {
                vector endPoint = closestRoad[closestRoad.Count() - 1];
                array<vector> nextRoad = FindConnectedRoad(endPoint, closestRoad);
                
                if (nextRoad && nextRoad.Count() > 0)
                {
                    for (int i = 0; i < nextRoad.Count() && segmentsAdded < m_iPathPredictionSegments; i++)
                    {
                        vector pathPoint = nextRoad[i];
                        // Raise height for visibility
                        pathPoint[1] += m_fPathHeight;
                        m_PredictedPath.Insert(pathPoint);
                        segmentsAdded++;
                    }
                }
            }
        }
        else
        {
            // Reverse direction (towards start of road)
            for (int i = closestIndex; i >= 0 && segmentsAdded < m_iPathPredictionSegments; i--)
            {
                vector pathPoint = closestRoad[i];
                // Raise height for visibility
                pathPoint[1] += m_fPathHeight;
                m_PredictedPath.Insert(pathPoint);
                segmentsAdded++;
            }
            
            // If we need more segments, look for connected roads
            if (segmentsAdded < m_iPathPredictionSegments && closestRoad.Count() > 0)
            {
                vector startPoint = closestRoad[0];
                array<vector> nextRoad = FindConnectedRoad(startPoint, closestRoad);
                
                if (nextRoad && nextRoad.Count() > 0)
                {
                    for (int i = nextRoad.Count() - 1; i >= 0 && segmentsAdded < m_iPathPredictionSegments; i--)
                    {
                        vector pathPoint = nextRoad[i];
                        // Raise height for visibility
                        pathPoint[1] += m_fPathHeight;
                        m_PredictedPath.Insert(pathPoint);
                        segmentsAdded++;
                    }
                }
            }
        }
        
        PrintFormat("SCR_EnhancedRoadVisualizer: Generated path prediction with %1 points", m_PredictedPath.Count());
    }
    
    //------------------------------------------------------------------------------------------------
    // Find a road connected to the given point (excluding the provided road)
    array<vector> FindConnectedRoad(vector connectionPoint, array<vector> excludeRoad)
    {
        // Find roads that connect to this point (with some tolerance)
        foreach (array<vector> road : m_RoadPaths)
        {
            // Skip the road we're already on
            if (road == excludeRoad)
                continue;
                
            // Check if this road connects at the start or end
            if (road.Count() > 0)
            {
                vector start = road[0];
                vector end = road[road.Count() - 1];
                
                // Check for connection with some tolerance
                float tolerance = 3.0;
                if (vector.Distance(start, connectionPoint) < tolerance ||
                    vector.Distance(end, connectionPoint) < tolerance)
                {
                    return road;
                }
            }
        }
        
        return null;
    }
    
    //------------------------------------------------------------------------------------------------
    // Creates all the visual elements for roads, junctions and paths
    void CreateVisualizations()
    {
        // Draw road centerlines
        if (m_bDrawRoadLines)
        {
            DrawRoadCenterlines();
        }
        
        // Draw road nodes/points
        if (m_bDrawRoadNodes)
        {
            DrawRoadNodes();
        }
        
        // Draw junctions
        if (m_bDrawJunctions)
        {
            DrawJunctions();
        }
        
        // Draw path prediction
        if (m_bDrawPathPrediction && !m_PredictedPath.IsEmpty())
        {
            DrawPathPrediction();
        }
        
        // Draw potential path options
        if (m_bDrawPathOptions)
        {
            DrawPathOptions();
        }
        
        Print("SCR_EnhancedRoadVisualizer: Created " + m_DebugShapes.Count() + " visualization shapes");
    }
    
    //------------------------------------------------------------------------------------------------
    // Draw road centerlines
    void DrawRoadCenterlines()
    {
        foreach (array<vector> road : m_RoadPaths)
        {
            if (road.Count() < 2)
                continue;
                
            // Create a polyline for the entire road
            int numPoints = road.Count();
            array<vector> linePoints = {};
            
            float spacing = Math.Max(1.0, m_fRoadLineDensity); // Spacing between visualization points
            float currentDist = 0;
            
            for (int i = 0; i < numPoints - 1; i++)
            {
                vector start = road[i];
                vector end = road[i + 1];
                vector segment = end - start;
                float segmentLength = segment.Length();
                
                // Skip if segmentLength is too small
                if (segmentLength < 0.1)
                    continue;
                    
                segment = segment / segmentLength; // Normalize
                
                // Add points along the segment at specified spacing
                while (currentDist < segmentLength)
                {
                    vector point = start + segment * currentDist;
                    // Raise slightly above terrain for visibility
                    point[1] += 0.5;
                    linePoints.Insert(point);
                    currentDist += spacing;
                }
                
                // Reset for next segment
                currentDist -= segmentLength;
            }
            
            // Always add the last point
            if (numPoints > 0)
            {
                vector lastPoint = road[numPoints - 1];
                lastPoint[1] += 0.5;
                linePoints.Insert(lastPoint);
            }
            
            // Create the road line
            if (linePoints.Count() >= 2)
            {
                Shape roadLine = Shape.CreateLines(
                    m_RoadSegmentColor.PackToInt(),
                    ShapeFlags.VISIBLE | ShapeFlags.NOOUTLINE | ShapeFlags.WIREFRAME,
                    linePoints,
                    linePoints.Count()
                );
                
                if (roadLine)
                    m_DebugShapes.Insert(roadLine);
            }
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Draw road nodes/points
    void DrawRoadNodes()
    {
        foreach (array<vector> road : m_RoadPaths)
        {
            // Only visualize a subset of points to avoid clutter
            int step = Math.Max(1, road.Count() / 10);
            
            for (int i = 0; i < road.Count(); i += step)
            {
                vector point = road[i];
                
                // Draw a small sphere at each node
                Shape nodeMarker = Shape.CreateSphere(
                    m_NodeColor.PackToInt(),
                    ShapeFlags.VISIBLE | ShapeFlags.NOOUTLINE,
                    point,
                    m_fNodeSize
                );
                
                if (nodeMarker)
                    m_DebugShapes.Insert(nodeMarker);
            }
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Draw junction markers
    void DrawJunctions()
    {
        foreach (vector junctionPoint : m_JunctionPoints)
        {
            // Draw a larger sphere at each junction
            Shape junctionMarker = Shape.CreateSphere(
                m_JunctionColor.PackToInt(),
                ShapeFlags.VISIBLE | ShapeFlags.NOOUTLINE,
                junctionPoint,
                m_fJunctionSize
            );
            
            if (junctionMarker)
                m_DebugShapes.Insert(junctionMarker);
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Draw path prediction
    void DrawPathPrediction()
    {
        if (m_PredictedPath.Count() < 2)
            return;
            
        // Create a line for the predicted path
        Shape pathLine = Shape.CreateLines(
            m_PathColor.PackToInt(),
            ShapeFlags.VISIBLE | ShapeFlags.NOOUTLINE | ShapeFlags.WIREFRAME,
            m_PredictedPath,
            m_PredictedPath.Count()
        );
        
        if (pathLine)
            m_DebugShapes.Insert(pathLine);
            
        // Add arrow at the end to show direction
        if (m_PredictedPath.Count() >= 2)
        {
            vector endPos = m_PredictedPath[m_PredictedPath.Count() - 1];
            vector prevPos = m_PredictedPath[m_PredictedPath.Count() - 2];
            vector dir = endPos - prevPos;
            dir.Normalize();
            
            // Create arrow head
            float arrowSize = 3.0;
            vector arrowLeft = endPos - dir * arrowSize + vector.CrossProduct(dir, Vector(0, 1, 0)) * arrowSize * 0.5;
            vector arrowRight = endPos - dir * arrowSize - vector.CrossProduct(dir, Vector(0, 1, 0)) * arrowSize * 0.5;
            
            array<vector> arrowPoints = {endPos, arrowLeft, endPos, arrowRight};
            
            Shape arrowShape = Shape.CreateLines(
                m_PathColor.PackToInt(),
                ShapeFlags.VISIBLE | ShapeFlags.NOOUTLINE | ShapeFlags.WIREFRAME,
                arrowPoints,
                arrowPoints.Count()
            );
            
            if (arrowShape)
                m_DebugShapes.Insert(arrowShape);
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Draw potential path options at junctions
    void DrawPathOptions()
    {
        // For each junction, draw potential paths
        foreach (vector junction : m_JunctionPoints)
        {
            // Find all roads connected to this junction
            array<array<vector>> connectedRoads = {};
            
            foreach (array<vector> road : m_RoadPaths)
            {
                if (road.Count() < 2)
                    continue;
                    
                // Check if road starts or ends at this junction
                vector start = road[0];
                vector end = road[road.Count() - 1];
                
                float tolerance = 3.0;
                if (vector.Distance(start, junction) < tolerance ||
                    vector.Distance(end, junction) < tolerance)
                {
                    connectedRoads.Insert(road);
                }
            }
            
            // Draw a segment of each connected road with a different color
            int colorHue = 0;
            foreach (array<vector> road : connectedRoads)
            {
                // Determine which end connects to the junction
                bool startsAtJunction = vector.Distance(road[0], junction) < 5.0;
                
                // Get a segment of the road
                array<vector> segment = {};
                int segmentLength = Math.Min(5, road.Count());
                
                if (startsAtJunction)
                {
                    for (int i = 0; i < segmentLength; i++)
                    {
                        segment.Insert(road[i]);
                    }
                }
                else
                {
                    for (int i = road.Count() - 1; i >= road.Count() - segmentLength && i >= 0; i--)
                    {
                        segment.Insert(road[i]);
                    }
                }
                
                // Skip if segment is too short
                if (segment.Count() < 2)
                    continue;
                
                // Create a unique color for this path option
                int optionColor = HSVToRGB(colorHue, 1.0, 1.0);
                colorHue = (colorHue + 60) % 360; // Increment hue by 60 degrees
                
                // Draw the path option
                Shape optionLine = Shape.CreateLines(
                    optionColor,
                    ShapeFlags.VISIBLE | ShapeFlags.NOOUTLINE | ShapeFlags.WIREFRAME,
                    segment,
                    segment.Count()
                );
                
                if (optionLine)
                    m_DebugShapes.Insert(optionLine);
            }
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Convert HSV color to RGB integer
    int HSVToRGB(float h, float s, float v)
    {
        // Normalize h to 0-1 range
        h = h / 360.0;
        
        // HSV to RGB conversion
        float c = v * s;
        float x = c * (1 - Math.AbsFloat(Math.Fmod(h * 6, 2) - 1));
        float m = v - c;
        
        float r, g, b;
        
        if (h < 1.0/6.0) { r = c; g = x; b = 0; }
        else if (h < 2.0/6.0) { r = x; g = c; b = 0; }
        else if (h < 3.0/6.0) { r = 0; g = c; b = x; }
        else if (h < 4.0/6.0) { r = 0; g = x; b = c; }
        else if (h < 5.0/6.0) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        
        r = (r + m) * 255;
        g = (g + m) * 255;
        b = (b + m) * 255;
        
        int rgb = ((int)r << 16) | ((int)g << 8) | (int)b;
        return rgb | 0xFF000000; // Add alpha channel
    }
    
    //------------------------------------------------------------------------------------------------
    // Clear all visualization shapes
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
    // Find the closest road segment to a position
    array<vector> GetClosestRoadSegment(vector position)
    {
        if (m_RoadPaths.IsEmpty())
            return null;
            
        float minDistance = float.MAX;
        int bestRoadIndex = -1;
        
        // Find the closest road
        for (int i = 0; i < m_RoadPaths.Count(); i++)
        {
            array<vector> road = m_RoadPaths[i];
            if (road.IsEmpty())
                continue;
                
            // Check distance to multiple points on this road
            float roadMinDist = float.MAX;
            
            // Sample a few points (start, middle, end)
            array<vector> samplePoints = {road[0], road[road.Count() - 1]};
            
            // Add a middle point if available
            if (road.Count() > 2)
                samplePoints.Insert(road[road.Count() / 2]);
                
            // Find minimum distance to any of these points
            foreach (vector point : samplePoints)
            {
                float dist = vector.Distance(position, point);
                if (dist < roadMinDist)
                    roadMinDist = dist;
            }
            
            // Update best road if this is closer
            if (roadMinDist < minDistance)
            {
                minDistance = roadMinDist;
                bestRoadIndex = i;
            }
        }
        
        // Return the closest road if found
        if (bestRoadIndex >= 0)
        {
            return m_RoadPaths[bestRoadIndex];
        }
        
        return null;
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
            if (moveDistance > 10.0) // Only update if moved significantly
            {
                ExtractAndVisualize();
            }
            else if (m_bDrawPathPrediction && m_ConnectedVehicle)
            {
                // Just update the path prediction without redoing road extraction
                ClearShapes();
                GeneratePathPrediction();
                CreateVisualizations();
            }
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API to manually trigger update
    void UpdateVisualization()
    {
        ExtractAndVisualize();
    }
    
    //------------------------------------------------------------------------------------------------
    // Toggle a specific visualization feature
    void ToggleFeature(string featureName, bool enable)
    {
        switch (featureName)
        {
            case "lines":
                m_bDrawRoadLines = enable;
                break;
            case "nodes":
                m_bDrawRoadNodes = enable;
                break;
            case "junctions":
                m_bDrawJunctions = enable;
                break;
            case "path":
                m_bDrawPathPrediction = enable;
                break;
            case "options":
                m_bDrawPathOptions = enable;
                break;
            default:
                return;
        }
        
        // Update visualization
        ExtractAndVisualize();
    }
    
    //------------------------------------------------------------------------------------------------
    // Set a target vehicle to follow for path prediction
    void SetTargetVehicle(Vehicle vehicle)
    {
        m_ConnectedVehicle = vehicle;
        if (vehicle)
        {
            Print("SCR_EnhancedRoadVisualizer: Now targeting vehicle " + vehicle.GetName());
        }
        else
        {
            Print("SCR_EnhancedRoadVisualizer: Cleared target vehicle");
        }
        
        // Update visualization
        ExtractAndVisualize();
    }
}