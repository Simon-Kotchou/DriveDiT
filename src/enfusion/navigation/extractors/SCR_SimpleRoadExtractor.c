// Optimized Road Network Extractor - Processes roads in chunks to avoid memory issues
[ComponentEditorProps(category: "GameScripted/AI/Navigation", description: "Road Network Extractor")]
class SCR_RoadNetworkExtractorClass: ScriptComponentClass
{
}

class SCR_RoadNetworkExtractor: ScriptComponent
{
    [Attribute("Missions/Roads.log", UIWidgets.EditBox, "Output file path for road data (relative to game directory)")]
    protected string m_sOutputFilePath;
    
    [Attribute("$profile:Roads/RoadPaths.log", UIWidgets.EditBox, "Alternative output path if main path fails")]
    protected string m_sAlternativeOutputPath;
    
    [Attribute("$temp:Roads.log", UIWidgets.EditBox, "Last resort output path")]
    protected string m_sLastResortPath;
    
    [Attribute("1", UIWidgets.CheckBox, "Auto-extract road network on startup")]
    protected bool m_bAutoExtract;
    
    [Attribute("1", UIWidgets.CheckBox, "Draw debug visualization of roads")]
    protected bool m_bDebugVisualize;
    
    [Attribute("1000", UIWidgets.Slider, "Size of each chunk for road extraction (in meters)", "100 5000 100")]
    protected float m_fChunkSize;

    [Attribute("100", UIWidgets.Slider, "Maximum roads to process per chunk", "10 1000 10")]
    protected int m_iMaxRoadsPerChunk;
    
    [Attribute("1000", UIWidgets.Slider, "Maximum debug visualization points", "100 5000 100")]
    protected int m_iMaxDebugPoints;

    protected SCR_AIWorld m_AIWorld;
    protected RoadNetworkManager m_RoadNetworkManager;
    protected FileHandle m_OutputFile;
    protected ref array<vector> m_DebugPoints;
    protected string m_FinalOutputPath;
    
    // Store a subset of road data in memory for API access
    protected ref array<ref array<vector>> m_StoredRoadPaths;
    protected int m_MaxStoredRoads = 100;  // Limit to control memory usage
    
    // Grid parameters for chunk-based extraction
    protected vector m_WorldOrigin;
    protected vector m_WorldMax;
    protected int m_CurrentChunkX;
    protected int m_CurrentChunkY;
    protected int m_TotalChunksX;
    protected int m_TotalChunksY;
    protected int m_TotalRoadsProcessed;
    
    //------------------------------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);
        
        m_DebugPoints = new array<vector>();
        m_StoredRoadPaths = new array<ref array<vector>>();
        m_FinalOutputPath = "";
        
        Print("SCR_RoadNetworkExtractor: Initializing...", LogLevel.NORMAL);
        
        // Use callqueue to ensure world is fully loaded
        if (m_bAutoExtract)
        {
            GetGame().GetCallqueue().CallLater(BeginRoadExtraction, 1000, false);
        }
    }
    
    //------------------------------------------------------------------------------------------------
    // Try to ensure directory exists for a file path
    protected bool EnsureDirectoryExists(string filePath)
    {
        int lastSlash = filePath.LastIndexOf("/");
        if (lastSlash != -1)
        {
            string directory = filePath.Substring(0, lastSlash);
            // Try to create the directory and log the result
            bool success = FileIO.MakeDirectory(directory);
            Print("SCR_RoadNetworkExtractor: Directory creation for " + directory + ": " + success);
            return success;
        }
        return true; // No directory needed
    }
    
    //------------------------------------------------------------------------------------------------
    // Try to open a file for writing with proper logging
    protected FileHandle TryOpenFile(string filePath)
    {
        // Log the attempt
        Print("SCR_RoadNetworkExtractor: Attempting to open file at " + filePath);
        
        // Ensure directory exists
        EnsureDirectoryExists(filePath);
        
        // Try to open the file
        FileHandle file = FileIO.OpenFile(filePath, FileMode.WRITE);
        
        // Log result
        if (file)
        {
            Print("SCR_RoadNetworkExtractor: Successfully opened file at " + filePath);
            m_FinalOutputPath = filePath;
            return file;
        }
        else
        {
            Print("SCR_RoadNetworkExtractor: Failed to open file at " + filePath, LogLevel.WARNING);
            return null;
        }
    }
    
    //------------------------------------------------------------------------------------------------
    bool InitializeRoadNetworkManager()
    {
        // Get the AI world instance
        m_AIWorld = SCR_AIWorld.Cast(GetGame().GetAIWorld());
        
        if (!m_AIWorld)
        {
            Print("SCR_RoadNetworkExtractor: ERROR - AIWorld not found!", LogLevel.ERROR);
            return false;
        }
        
        // Get the road network manager
        m_RoadNetworkManager = m_AIWorld.GetRoadNetworkManager();
        if (!m_RoadNetworkManager)
        {
            Print("SCR_RoadNetworkExtractor: ERROR - RoadNetworkManager not available.", LogLevel.ERROR);
            return false;
        }
        
        Print("SCR_RoadNetworkExtractor: Successfully connected to road network manager.");
        return true;
    }
    
    //------------------------------------------------------------------------------------------------
    // Begin the extraction process by determining the map boundaries and setting up chunks
    void BeginRoadExtraction()
    {
        Print("SCR_RoadNetworkExtractor: Starting road network extraction...");
        
        // Initialize road network manager
        if (!InitializeRoadNetworkManager())
            return;
        
        // Try each path in sequence until one works
        m_OutputFile = TryOpenFile(m_sOutputFilePath);
        
        if (!m_OutputFile)
            m_OutputFile = TryOpenFile(m_sAlternativeOutputPath);
            
        if (!m_OutputFile)
            m_OutputFile = TryOpenFile(m_sLastResortPath);
            
        if (!m_OutputFile)
        {
            // Try a direct game folder path as a last resort
            m_OutputFile = TryOpenFile("roads_output.log");
        }
        
        if (!m_OutputFile)
        {
            Print("SCR_RoadNetworkExtractor: WARNING - All file output attempts failed. Will store data in memory only.", LogLevel.WARNING);
        }
        
        // Write file header if file is open
        if (m_OutputFile)
        {
            m_OutputFile.WriteLine("# Road Network Data - Extracted by SCR_RoadNetworkExtractor");
            m_OutputFile.WriteLine("# Format: Road[ID] - PointCount: (x,y,z) (x,y,z) ...");
            m_OutputFile.WriteLine("# Processed in chunks to optimize memory usage");
            m_OutputFile.WriteLine("# ----------------------------------------");
        }
        
        // Get world size - using a more reasonable approach than before
        // Note: In practice, you might want to query actual terrain size from the engine
        m_WorldOrigin = Vector(-20000, 0, -20000); // Assuming a reasonable map size
        m_WorldMax = Vector(20000, 0, 20000);
        
        // Calculate how many chunks we need
        m_TotalChunksX = Math.Ceil((m_WorldMax[0] - m_WorldOrigin[0]) / m_fChunkSize);
        m_TotalChunksY = Math.Ceil((m_WorldMax[2] - m_WorldOrigin[2]) / m_fChunkSize);
        
        // Initialize chunk processing
        m_CurrentChunkX = 0;
        m_CurrentChunkY = 0;
        m_TotalRoadsProcessed = 0;
        
        // Start processing chunks
        PrintFormat("SCR_RoadNetworkExtractor: Map divided into %1x%2 chunks (%3m each)", 
            m_TotalChunksX, m_TotalChunksY, m_fChunkSize);
        
        // Process first chunk immediately, then schedule the rest
        ProcessNextChunk();
    }
    
    //------------------------------------------------------------------------------------------------
    // Process the current chunk and schedule the next one
    void ProcessNextChunk()
    {
        // Check if we've completed all chunks
        if (m_CurrentChunkX >= m_TotalChunksX)
        {
            FinishRoadExtraction();
            return;
        }
        
        // Calculate chunk boundaries
        vector chunkMin, chunkMax;
        chunkMin[0] = m_WorldOrigin[0] + (m_CurrentChunkX * m_fChunkSize);
        chunkMin[2] = m_WorldOrigin[2] + (m_CurrentChunkY * m_fChunkSize);
        chunkMin[1] = -1000; // Height range
        
        chunkMax[0] = chunkMin[0] + m_fChunkSize;
        chunkMax[2] = chunkMin[2] + m_fChunkSize;
        chunkMax[1] = 1000; // Height range
        
        PrintFormat("SCR_RoadNetworkExtractor: Processing chunk [%1,%2] of [%3,%4]", 
            m_CurrentChunkX + 1, m_CurrentChunkY + 1, m_TotalChunksX, m_TotalChunksY);
        
        // Extract roads in this chunk
        ExtractRoadsInChunk(chunkMin, chunkMax);
        
        // Move to next chunk
        m_CurrentChunkY++;
        if (m_CurrentChunkY >= m_TotalChunksY)
        {
            m_CurrentChunkY = 0;
            m_CurrentChunkX++;
        }
        
        // Schedule next chunk processing with a small delay to allow frame to complete
        GetGame().GetCallqueue().CallLater(ProcessNextChunk, 50, false);
    }
    
    //------------------------------------------------------------------------------------------------
    // Extract road data for a specific chunk
    void ExtractRoadsInChunk(vector chunkMin, vector chunkMax)
    {
        // Array to temporarily hold roads in this chunk only
        array<BaseRoad> chunkRoads = {};
        
        // Get roads in this AABB chunk
        int roadCount = m_RoadNetworkManager.GetRoadsInAABB(chunkMin, chunkMax, chunkRoads);
        
        // Limit the number of roads processed if needed to avoid memory issues
        if (roadCount > m_iMaxRoadsPerChunk)
        {
            PrintFormat("SCR_RoadNetworkExtractor: Limiting chunk from %1 to %2 roads", 
                roadCount, m_iMaxRoadsPerChunk);
            roadCount = m_iMaxRoadsPerChunk;
        }
        
        if (roadCount <= 0)
            return;
        
        // Process each road in this chunk
        for (int roadIndex = 0; roadIndex < roadCount; roadIndex++)
        {
            BaseRoad road = chunkRoads[roadIndex];
            
            // Temporary array for points - only used briefly and then cleared
            array<vector> roadPoints = {};
            road.GetPoints(roadPoints);
            
            if (roadPoints.IsEmpty())
                continue;
            
            // Log the road points
            string roadLine = string.Format("Road[%1] - %2 points: ", m_TotalRoadsProcessed, roadPoints.Count());
            
            // Only add a subset of points to visualization to save memory
            int visualizationStep = Math.Max(1, roadPoints.Count() / 10); // At most 10 points per road
            
            // Append point coordinates
            for (int i = 0; i < roadPoints.Count(); i++)
            {
                vector p = roadPoints[i];
                roadLine = roadLine + string.Format("(%1,%2,%3) ", p[0], p[1], p[2]);
                
                // Add selected points to visualization array
                if (m_bDebugVisualize && (i % visualizationStep == 0) && m_DebugPoints.Count() < m_iMaxDebugPoints)
                {
                    m_DebugPoints.Insert(p);
                }
            }
            
            // Write to file
            if (m_OutputFile) 
            {
                m_OutputFile.WriteLine(roadLine);
            }
            
            // Add points to in-memory storage if needed
            if (!m_OutputFile && m_StoredRoadPaths.Count() < m_MaxStoredRoads)
            {
                ref array<vector> roadPath = new array<vector>();
                roadPath.Copy(roadPoints);
                m_StoredRoadPaths.Insert(roadPath);
            }
            
            // Print first and last point to console as a brief confirmation
            vector start = roadPoints[0];
            vector end = roadPoints[roadPoints.Count() - 1];
            PrintFormat("Road[%1]: %2 points (start=%3, end=%4)", 
                m_TotalRoadsProcessed, roadPoints.Count(), start, end);
            
            m_TotalRoadsProcessed++;
            
            // Clear points array to free memory after processing each road
            roadPoints.Clear();
        }
        
        // Clear the chunk roads array to free memory
        chunkRoads.Clear();
    }
    
    //------------------------------------------------------------------------------------------------
    // Finish the road extraction process
    void FinishRoadExtraction()
    {
        // Write footer to file
        if (m_OutputFile)
        {
            m_OutputFile.WriteLine("# ----------------------------------------");
            m_OutputFile.WriteLine(string.Format("# Total Roads Processed: %1", m_TotalRoadsProcessed));
            m_OutputFile.Close();
            m_OutputFile = null;
            
            Print("SCR_RoadNetworkExtractor: Road paths saved to " + m_FinalOutputPath, LogLevel.NORMAL);
        }
        else
        {
            Print("SCR_RoadNetworkExtractor: Note: Data was stored in memory only (no file output).", LogLevel.WARNING);
            Print(string.Format("SCR_RoadNetworkExtractor: %1 roads stored in memory for runtime access.", m_StoredRoadPaths.Count()));
        }
        
        // Create debug visualization if enabled
        if (m_bDebugVisualize && !m_DebugPoints.IsEmpty())
        {
            CreateRoadVisualization();
        }
        
        Print(string.Format("SCR_RoadNetworkExtractor: Road extraction complete! Processed %1 roads.", m_TotalRoadsProcessed), LogLevel.NORMAL);
    }
    
    //------------------------------------------------------------------------------------------------
    // Create visualization points for roads - now using the optimized subset
    void CreateRoadVisualization()
    {
        int pointCount = m_DebugPoints.Count();
        Print("SCR_RoadNetworkExtractor: Creating visualization for " + pointCount + " road points");
        
        if (pointCount > m_iMaxDebugPoints)
        {
            Print(string.Format("SCR_RoadNetworkExtractor: Limiting visualization from %1 to %2 points", 
                pointCount, m_iMaxDebugPoints));
            pointCount = m_iMaxDebugPoints;
        }
        
        // Draw points
        for (int i = 0; i < pointCount; i++)
        {
            // Create a sphere at each road point
            Shape sphere = Shape.CreateSphere(
                COLOR_GREEN, 
                ShapeFlags.VISIBLE | ShapeFlags.NOOUTLINE, 
                m_DebugPoints[i], 
                0.5 // Radius
            );
        }
        
        Print("SCR_RoadNetworkExtractor: Debug visualization created");
        
        // Clear debug points array to free memory
        m_DebugPoints.Clear();
    }
    
    //------------------------------------------------------------------------------------------------
    // Get roads near a specific position - useful utility function
    array<BaseRoad> GetRoadsNearPosition(vector position, float radius)
    {
        if (!m_RoadNetworkManager)
            InitializeRoadNetworkManager();
            
        if (!m_RoadNetworkManager)
            return null;
        
        array<BaseRoad> nearbyRoads = {};
        
        // Create a small AABB around the position
        vector min = position - Vector(radius, radius, radius);
        vector max = position + Vector(radius, radius, radius);
        
        // Get roads in this small area
        m_RoadNetworkManager.GetRoadsInAABB(min, max, nearbyRoads);
        
        return nearbyRoads;
    }
    
    //------------------------------------------------------------------------------------------------
    // Get the closest road point to a position
    bool GetClosestRoadPoint(vector position, float searchRadius, out vector closestPoint)
    {
        array<BaseRoad> nearbyRoads = GetRoadsNearPosition(position, searchRadius);
        
        if (!nearbyRoads || nearbyRoads.IsEmpty())
            return false;
        
        float minDistance = float.MAX;
        
        // Check each road
        foreach (BaseRoad road : nearbyRoads)
        {
            array<vector> roadPoints = {};
            road.GetPoints(roadPoints);
            
            // Find closest point on this road
            foreach (vector point : roadPoints)
            {
                float distance = vector.Distance(position, point);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closestPoint = point;
                }
            }
            
            // Clear points to free memory
            roadPoints.Clear();
        }
        
        return minDistance < float.MAX;
    }
    
    //------------------------------------------------------------------------------------------------
    // Get a road path by index from in-memory storage
    array<vector> GetRoadPathByIndex(int index)
    {
        if (index < 0 || index >= m_StoredRoadPaths.Count())
            return null;
            
        return m_StoredRoadPaths[index];
    }
    
    //------------------------------------------------------------------------------------------------
    // Get the number of roads stored in memory
    int GetStoredRoadCount()
    {
        return m_StoredRoadPaths.Count();
    }
    
    //------------------------------------------------------------------------------------------------
    // Check if extraction is complete
    bool IsExtractionComplete()
    {
        return m_CurrentChunkX >= m_TotalChunksX;
    }
    
    //------------------------------------------------------------------------------------------------
    // Manually trigger a save to a specific path (can be called after extraction completes)
    bool SaveRoadDataToFile(string customPath)
    {
        if (!IsExtractionComplete() || m_StoredRoadPaths.IsEmpty())
        {
            Print("SCR_RoadNetworkExtractor: Cannot save data - either extraction not complete or no data available", LogLevel.ERROR);
            return false;
        }
        
        FileHandle file = FileIO.OpenFile(customPath, FileMode.WRITE);
        if (!file)
        {
            Print("SCR_RoadNetworkExtractor: Failed to open custom save path: " + customPath, LogLevel.ERROR);
            return false;
        }
        
        // Write header
        file.WriteLine("# Road Network Data - Extracted by SCR_RoadNetworkExtractor");
        file.WriteLine("# Format: Road[ID] - PointCount: (x,y,z) (x,y,z) ...");
        file.WriteLine("# ----------------------------------------");
        
        // Write each stored road
        for (int i = 0; i < m_StoredRoadPaths.Count(); i++)
        {
            array<vector> points = m_StoredRoadPaths[i];
            if (!points || points.IsEmpty())
                continue;
                
            string roadLine = string.Format("Road[%1] - %2 points: ", i, points.Count());
            
            foreach (vector p : points)
            {
                roadLine = roadLine + string.Format("(%1,%2,%3) ", p[0], p[1], p[2]);
            }
            
            file.WriteLine(roadLine);
        }
        
        // Write footer
        file.WriteLine("# ----------------------------------------");
        file.WriteLine(string.Format("# Total Roads Saved: %1", m_StoredRoadPaths.Count()));
        file.Close();
        
        Print("SCR_RoadNetworkExtractor: Successfully saved road data to: " + customPath, LogLevel.NORMAL);
        return true;
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API for manual triggering from other scripts
    //------------------------------------------------------------------------------------------------
    void ExtractRoads()
    {
        BeginRoadExtraction();
    }
    
    //------------------------------------------------------------------------------------------------
    // Public API to get road data for specific vehicle/entity - useful for AI driving logic
    //------------------------------------------------------------------------------------------------
    bool GetRoadDataForVehicle(IEntity vehicle, float radius, out array<vector> roadPath)
    {
        if (!vehicle)
            return false;
            
        vector vehiclePos = vehicle.GetOrigin();
        array<BaseRoad> nearbyRoads = GetRoadsNearPosition(vehiclePos, radius);
        
        if (!nearbyRoads || nearbyRoads.IsEmpty())
            return false;
            
        // Find closest road
        float minDistance = float.MAX;
        BaseRoad closestRoad = null;
        
        foreach (BaseRoad road : nearbyRoads)
        {
            array<vector> points = {};
            road.GetPoints(points);
            
            if (points.IsEmpty())
                continue;
                
            // Check first and last point to estimate distance
            float distStart = vector.Distance(vehiclePos, points[0]);
            float distEnd = vector.Distance(vehiclePos, points[points.Count() - 1]);
            float minDist = Math.Min(distStart, distEnd);
            
            if (minDist < minDistance)
            {
                minDistance = minDist;
                closestRoad = road;
            }
            
            points.Clear();
        }
        
        if (!closestRoad)
            return false;
            
        // Get points for closest road
        roadPath = new array<vector>();
        closestRoad.GetPoints(roadPath);
        return !roadPath.IsEmpty();
    }
}