// ============================================================================
// SCR_DepthRaycaster - Grid-Based Depth Capture via Raycasting
// ============================================================================
//
// Captures depth maps using raycasting for world model training data.
// Designed for Self-Forcing++ and Depth Anything V3 distillation pipelines.
//
// FEATURES:
//   - Configurable grid resolution (e.g., 64x48, 128x96, 256x192)
//   - Pre-computed ray directions based on camera FOV
//   - Optimized batch processing with temporal caching
//   - Sky region detection and skipping
//   - Validity mask for incomplete rays
//   - Integration with SCR_MLDataCollector
//
// OUTPUT FORMAT:
//   - Float array of depth values (meters)
//   - Boolean validity mask
//   - Normalized depth map (0-1 range)
//
// USAGE:
//   SCR_DepthRaycaster raycaster = new SCR_DepthRaycaster();
//   raycaster.Initialize(128, 96, 90.0);
//   raycaster.CaptureDepth(cameraTransform, world);
//   array<float> depths = raycaster.GetDepthBuffer();
//
// ============================================================================

// Layer mask constants for depth tracing
// Based on Enfusion collision layer system
const int DEPTH_LAYER_TERRAIN = 1;
const int DEPTH_LAYER_STATIC = 2;
const int DEPTH_LAYER_DYNAMIC = 4;
const int DEPTH_LAYER_VEHICLE = 8;
const int DEPTH_LAYER_CHARACTER = 16;
const int DEPTH_LAYER_PROP = 32;
const int DEPTH_LAYER_BUILDING = 64;
const int DEPTH_LAYER_FOLIAGE = 128;

// Combined layer mask for depth capture (excludes some layers for performance)
const int DEPTH_TRACE_LAYERS = DEPTH_LAYER_TERRAIN | DEPTH_LAYER_STATIC |
                                DEPTH_LAYER_VEHICLE | DEPTH_LAYER_BUILDING |
                                DEPTH_LAYER_PROP;

// ============================================================================
// SCR_DepthRaycasterClass - Component class definition
// ============================================================================
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Grid-based depth capture via raycasting for ML training")]
class SCR_DepthRaycasterClass: ScriptComponentClass
{
}

// ============================================================================
// SCR_DepthRaycaster - Main depth capture component
// ============================================================================
class SCR_DepthRaycaster: ScriptComponent
{
    // === CONFIGURATION ===
    [Attribute("128", UIWidgets.Slider, "Depth grid width (pixels)", "16 512 16")]
    protected int m_iGridWidth;

    [Attribute("96", UIWidgets.Slider, "Depth grid height (pixels)", "12 384 12")]
    protected int m_iGridHeight;

    [Attribute("75.0", UIWidgets.Slider, "Vertical FOV in degrees", "30 120 5")]
    protected float m_fVerticalFOV;

    [Attribute("1.333", UIWidgets.EditBox, "Aspect ratio (width/height)")]
    protected float m_fAspectRatio;

    [Attribute("0.5", UIWidgets.Slider, "Near plane distance (meters)", "0.1 5.0 0.1")]
    protected float m_fNearPlane;

    [Attribute("1000.0", UIWidgets.Slider, "Far plane / max depth (meters)", "100 5000 100")]
    protected float m_fFarPlane;

    [Attribute("1", UIWidgets.CheckBox, "Enable temporal caching")]
    protected bool m_bEnableTemporalCache;

    [Attribute("3", UIWidgets.Slider, "Temporal cache frames", "1 10 1")]
    protected int m_iTemporalCacheFrames;

    [Attribute("1", UIWidgets.CheckBox, "Skip sky regions (no hit)")]
    protected bool m_bSkipSkyRegions;

    [Attribute("0", UIWidgets.CheckBox, "Enable debug visualization")]
    protected bool m_bDebugVisualization;

    [Attribute("1", UIWidgets.CheckBox, "Use checkerboard pattern for optimization")]
    protected bool m_bCheckerboardOptimization;

    // === INTERNAL STATE ===
    protected bool m_bInitialized;
    protected int m_iTotalPixels;

    // Pre-computed ray directions (local camera space)
    protected ref array<vector> m_aRayDirections;

    // Depth buffer (current frame)
    protected ref array<float> m_aDepthBuffer;

    // Validity mask (true = valid depth, false = sky/no hit)
    protected ref array<bool> m_aValidityMask;

    // Normalized depth buffer (0-1 range)
    protected ref array<float> m_aNormalizedDepth;

    // Temporal cache for stability
    protected ref array<ref array<float>> m_aTemporalDepthCache;
    protected int m_iTemporalCacheIndex;

    // Performance tracking
    protected int m_iRaysCast;
    protected int m_iRaysHit;
    protected int m_iRaysSkipped;
    protected float m_fLastCaptureTimeMs;
    protected float m_fAverageCaptureTimeMs;
    protected int m_iCaptureCount;

    // Checkerboard state
    protected bool m_bCheckerboardPhase;

    // Reusable trace parameter (avoid allocation per ray)
    protected ref TraceParam m_TraceParam;

    // Statistics
    protected float m_fMinDepth;
    protected float m_fMaxDepth;
    protected float m_fMeanDepth;

    //------------------------------------------------------------------------
    // Component initialization
    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        m_bInitialized = false;
        m_iCaptureCount = 0;
        m_fAverageCaptureTimeMs = 0;
        m_bCheckerboardPhase = false;

        // Initialize with default configuration
        Initialize(m_iGridWidth, m_iGridHeight, m_fVerticalFOV);

        Print("[DepthRaycaster] Component initialized: " + m_iGridWidth + "x" + m_iGridHeight, LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Initialize depth capture system
    //------------------------------------------------------------------------
    bool Initialize(int gridWidth, int gridHeight, float verticalFOV)
    {
        if (gridWidth <= 0 || gridHeight <= 0 || verticalFOV <= 0)
        {
            Print("[DepthRaycaster] ERROR: Invalid parameters", LogLevel.ERROR);
            return false;
        }

        m_iGridWidth = gridWidth;
        m_iGridHeight = gridHeight;
        m_fVerticalFOV = verticalFOV;
        m_iTotalPixels = gridWidth * gridHeight;

        // Allocate buffers
        m_aRayDirections = new array<vector>();
        m_aDepthBuffer = new array<float>();
        m_aValidityMask = new array<bool>();
        m_aNormalizedDepth = new array<float>();

        // Pre-size arrays
        m_aRayDirections.Reserve(m_iTotalPixels);
        m_aDepthBuffer.Reserve(m_iTotalPixels);
        m_aValidityMask.Reserve(m_iTotalPixels);
        m_aNormalizedDepth.Reserve(m_iTotalPixels);

        // Initialize buffers with default values
        for (int i = 0; i < m_iTotalPixels; i++)
        {
            m_aRayDirections.Insert(vector.Zero);
            m_aDepthBuffer.Insert(m_fFarPlane);
            m_aValidityMask.Insert(false);
            m_aNormalizedDepth.Insert(1.0);
        }

        // Pre-compute ray directions
        ComputeRayDirections();

        // Initialize temporal cache if enabled
        if (m_bEnableTemporalCache)
        {
            InitializeTemporalCache();
        }

        // Create reusable trace parameter
        m_TraceParam = new TraceParam();
        m_TraceParam.LayerMask = DEPTH_TRACE_LAYERS;
        m_TraceParam.Flags = TraceFlags.WORLD | TraceFlags.ENTS;

        m_bInitialized = true;

        Print("[DepthRaycaster] Initialized: " + m_iTotalPixels + " pixels, FOV=" + m_fVerticalFOV, LogLevel.NORMAL);
        return true;
    }

    //------------------------------------------------------------------------
    // Pre-compute ray directions based on camera FOV
    // Rays are in local camera space (forward = +Z, right = +X, up = +Y)
    //------------------------------------------------------------------------
    protected void ComputeRayDirections()
    {
        // Convert FOV to radians
        float fovRad = m_fVerticalFOV * Math.DEG2RAD;
        float halfFovV = fovRad * 0.5;
        float halfFovH = Math.Atan(Math.Tan(halfFovV) * m_fAspectRatio);

        // Compute direction for each pixel
        for (int y = 0; y < m_iGridHeight; y++)
        {
            // Normalized y coordinate (-1 to 1, top to bottom)
            float ny = 1.0 - (2.0 * (y + 0.5) / m_iGridHeight);
            float angleV = ny * halfFovV;

            for (int x = 0; x < m_iGridWidth; x++)
            {
                // Normalized x coordinate (-1 to 1, left to right)
                float nx = (2.0 * (x + 0.5) / m_iGridWidth) - 1.0;
                float angleH = nx * halfFovH;

                // Compute ray direction in camera space
                // Camera looks along +Z, right is +X, up is +Y
                float dirX = Math.Tan(angleH);
                float dirY = Math.Tan(angleV);
                float dirZ = 1.0;

                // Normalize the direction vector
                vector dir = Vector(dirX, dirY, dirZ);
                dir.Normalize();

                int index = y * m_iGridWidth + x;
                m_aRayDirections[index] = dir;
            }
        }

        Print("[DepthRaycaster] Pre-computed " + m_iTotalPixels + " ray directions", LogLevel.VERBOSE);
    }

    //------------------------------------------------------------------------
    // Initialize temporal cache for depth smoothing
    //------------------------------------------------------------------------
    protected void InitializeTemporalCache()
    {
        m_aTemporalDepthCache = new array<ref array<float>>();
        m_iTemporalCacheIndex = 0;

        for (int f = 0; f < m_iTemporalCacheFrames; f++)
        {
            ref array<float> frame = new array<float>();
            frame.Reserve(m_iTotalPixels);
            for (int i = 0; i < m_iTotalPixels; i++)
            {
                frame.Insert(m_fFarPlane);
            }
            m_aTemporalDepthCache.Insert(frame);
        }

        Print("[DepthRaycaster] Temporal cache initialized: " + m_iTemporalCacheFrames + " frames", LogLevel.VERBOSE);
    }

    //------------------------------------------------------------------------
    // Main depth capture method
    // cameraTransform: 4x4 world transform matrix [right, up, forward, position]
    //------------------------------------------------------------------------
    bool CaptureDepth(vector cameraTransform[4], BaseWorld world)
    {
        if (!m_bInitialized)
        {
            Print("[DepthRaycaster] ERROR: Not initialized", LogLevel.ERROR);
            return false;
        }

        if (!world)
        {
            Print("[DepthRaycaster] ERROR: Invalid world", LogLevel.ERROR);
            return false;
        }

        float startTime = System.GetTickCount();

        // Reset statistics
        m_iRaysCast = 0;
        m_iRaysHit = 0;
        m_iRaysSkipped = 0;
        m_fMinDepth = m_fFarPlane;
        m_fMaxDepth = 0;
        float depthSum = 0;
        int validCount = 0;

        // Extract camera vectors
        vector camRight = cameraTransform[0];
        vector camUp = cameraTransform[1];
        vector camForward = cameraTransform[2];
        vector camPosition = cameraTransform[3];

        // Cast rays for each pixel
        for (int y = 0; y < m_iGridHeight; y++)
        {
            for (int x = 0; x < m_iGridWidth; x++)
            {
                int index = y * m_iGridWidth + x;

                // Checkerboard optimization: skip every other pixel, interpolate later
                if (m_bCheckerboardOptimization)
                {
                    bool isEvenPixel = ((x + y) % 2 == 0);
                    if (isEvenPixel != m_bCheckerboardPhase)
                    {
                        m_iRaysSkipped++;
                        continue;
                    }
                }

                // Get pre-computed local ray direction
                vector localDir = m_aRayDirections[index];

                // Transform to world space
                vector worldDir = localDir[0] * camRight +
                                  localDir[1] * camUp +
                                  localDir[2] * camForward;
                worldDir.Normalize();

                // Cast ray
                float depth = CastDepthRay(camPosition, worldDir, world);

                m_iRaysCast++;

                // Store result
                m_aDepthBuffer[index] = depth;

                if (depth < m_fFarPlane)
                {
                    m_aValidityMask[index] = true;
                    m_iRaysHit++;

                    // Update statistics
                    if (depth < m_fMinDepth)
                        m_fMinDepth = depth;
                    if (depth > m_fMaxDepth)
                        m_fMaxDepth = depth;
                    depthSum += depth;
                    validCount++;
                }
                else
                {
                    m_aValidityMask[index] = false;
                }
            }
        }

        // Interpolate checkerboard if enabled
        if (m_bCheckerboardOptimization)
        {
            InterpolateCheckerboard();
            m_bCheckerboardPhase = !m_bCheckerboardPhase;
        }

        // Apply temporal filtering if enabled
        if (m_bEnableTemporalCache)
        {
            ApplyTemporalFiltering();
        }

        // Compute normalized depth
        ComputeNormalizedDepth();

        // Update statistics
        if (validCount > 0)
            m_fMeanDepth = depthSum / validCount;
        else
            m_fMeanDepth = m_fFarPlane;

        // Performance tracking
        float endTime = System.GetTickCount();
        m_fLastCaptureTimeMs = endTime - startTime;
        m_iCaptureCount++;
        m_fAverageCaptureTimeMs = ((m_fAverageCaptureTimeMs * (m_iCaptureCount - 1)) + m_fLastCaptureTimeMs) / m_iCaptureCount;

        return true;
    }

    //------------------------------------------------------------------------
    // Cast a single depth ray and return distance
    //------------------------------------------------------------------------
    protected float CastDepthRay(vector origin, vector direction, BaseWorld world)
    {
        // Compute ray endpoints
        vector rayStart = origin + direction * m_fNearPlane;
        vector rayEnd = origin + direction * m_fFarPlane;

        // Configure trace
        m_TraceParam.Start = rayStart;
        m_TraceParam.End = rayEnd;
        m_TraceParam.TraceEnt = null;
        m_TraceParam.TraceDist = 0;

        // Execute trace
        float traceResult = world.TraceMove(m_TraceParam, null);

        if (traceResult < 1.0 && m_TraceParam.TraceEnt)
        {
            // Hit something - compute actual distance
            float distance = m_fNearPlane + (m_fFarPlane - m_fNearPlane) * traceResult;
            return distance;
        }

        // No hit - return far plane (sky)
        return m_fFarPlane;
    }

    //------------------------------------------------------------------------
    // Interpolate missing pixels from checkerboard pattern
    //------------------------------------------------------------------------
    protected void InterpolateCheckerboard()
    {
        for (int y = 0; y < m_iGridHeight; y++)
        {
            for (int x = 0; x < m_iGridWidth; x++)
            {
                int index = y * m_iGridWidth + x;

                // Check if this pixel was skipped
                bool isEvenPixel = ((x + y) % 2 == 0);
                if (isEvenPixel == m_bCheckerboardPhase)
                    continue; // This pixel was traced, skip

                // Interpolate from neighbors
                float sum = 0;
                int count = 0;
                bool anyValid = false;

                // Sample 4 neighbors
                int neighbors[4];
                neighbors[0] = (x > 0) ? index - 1 : -1;                    // Left
                neighbors[1] = (x < m_iGridWidth - 1) ? index + 1 : -1;     // Right
                neighbors[2] = (y > 0) ? index - m_iGridWidth : -1;         // Up
                neighbors[3] = (y < m_iGridHeight - 1) ? index + m_iGridWidth : -1; // Down

                for (int i = 0; i < 4; i++)
                {
                    int nIdx = neighbors[i];
                    if (nIdx >= 0 && nIdx < m_iTotalPixels)
                    {
                        if (m_aValidityMask[nIdx])
                        {
                            sum += m_aDepthBuffer[nIdx];
                            count++;
                            anyValid = true;
                        }
                    }
                }

                if (count > 0)
                {
                    m_aDepthBuffer[index] = sum / count;
                    m_aValidityMask[index] = anyValid;
                }
            }
        }
    }

    //------------------------------------------------------------------------
    // Apply temporal filtering for stability
    //------------------------------------------------------------------------
    protected void ApplyTemporalFiltering()
    {
        // Store current frame in cache
        ref array<float> currentCache = m_aTemporalDepthCache[m_iTemporalCacheIndex];
        for (int i = 0; i < m_iTotalPixels; i++)
        {
            currentCache[i] = m_aDepthBuffer[i];
        }

        // Advance cache index
        m_iTemporalCacheIndex = (m_iTemporalCacheIndex + 1) % m_iTemporalCacheFrames;

        // Only apply filtering if we have enough frames
        if (m_iCaptureCount < m_iTemporalCacheFrames)
            return;

        // Average across temporal cache
        for (int i = 0; i < m_iTotalPixels; i++)
        {
            if (!m_aValidityMask[i])
                continue;

            float sum = 0;
            int validFrames = 0;

            for (int f = 0; f < m_iTemporalCacheFrames; f++)
            {
                float cachedDepth = m_aTemporalDepthCache[f][i];
                if (cachedDepth < m_fFarPlane)
                {
                    sum += cachedDepth;
                    validFrames++;
                }
            }

            if (validFrames > 0)
            {
                m_aDepthBuffer[i] = sum / validFrames;
            }
        }
    }

    //------------------------------------------------------------------------
    // Compute normalized depth (0 = near, 1 = far)
    //------------------------------------------------------------------------
    protected void ComputeNormalizedDepth()
    {
        float range = m_fFarPlane - m_fNearPlane;
        if (range <= 0)
            range = 1.0;

        for (int i = 0; i < m_iTotalPixels; i++)
        {
            float depth = m_aDepthBuffer[i];
            float normalized = (depth - m_fNearPlane) / range;
            normalized = Math.Clamp(normalized, 0.0, 1.0);
            m_aNormalizedDepth[i] = normalized;
        }
    }

    //------------------------------------------------------------------------
    // Capture depth from current camera
    //------------------------------------------------------------------------
    bool CaptureFromCamera(CameraBase camera, BaseWorld world)
    {
        if (!camera)
        {
            Print("[DepthRaycaster] ERROR: Invalid camera", LogLevel.ERROR);
            return false;
        }

        // Get camera transform
        vector cameraTransform[4];
        camera.GetWorldCameraTransform(cameraTransform);

        // Update FOV from camera if available
        float cameraFOV = camera.GetVerticalFOV();
        if (cameraFOV > 0 && Math.AbsFloat(cameraFOV - m_fVerticalFOV) > 1.0)
        {
            m_fVerticalFOV = cameraFOV;
            ComputeRayDirections(); // Recompute rays with new FOV
        }

        // Update planes from camera
        float nearPlane = camera.GetNearPlane();
        float farPlane = camera.GetFarPlane();
        if (nearPlane > 0)
            m_fNearPlane = nearPlane;
        if (farPlane > 0)
            m_fFarPlane = Math.Min(farPlane, 2000.0); // Cap at 2km for performance

        return CaptureDepth(cameraTransform, world);
    }

    //------------------------------------------------------------------------
    // Capture depth from current game camera
    //------------------------------------------------------------------------
    bool CaptureFromCurrentCamera()
    {
        ArmaReforgerScripted game = GetGame();
        if (!game)
            return false;

        CameraManager camManager = game.GetCameraManager();
        if (!camManager)
            return false;

        CameraBase camera = camManager.CurrentCamera();
        if (!camera)
            return false;

        BaseWorld world = game.GetWorld();
        if (!world)
            return false;

        return CaptureFromCamera(camera, world);
    }

    //------------------------------------------------------------------------
    // Get raw depth buffer
    //------------------------------------------------------------------------
    array<float> GetDepthBuffer()
    {
        return m_aDepthBuffer;
    }

    //------------------------------------------------------------------------
    // Get normalized depth buffer (0-1)
    //------------------------------------------------------------------------
    array<float> GetNormalizedDepthBuffer()
    {
        return m_aNormalizedDepth;
    }

    //------------------------------------------------------------------------
    // Get validity mask
    //------------------------------------------------------------------------
    array<bool> GetValidityMask()
    {
        return m_aValidityMask;
    }

    //------------------------------------------------------------------------
    // Get depth at specific pixel
    //------------------------------------------------------------------------
    float GetDepthAt(int x, int y)
    {
        if (x < 0 || x >= m_iGridWidth || y < 0 || y >= m_iGridHeight)
            return m_fFarPlane;

        int index = y * m_iGridWidth + x;
        return m_aDepthBuffer[index];
    }

    //------------------------------------------------------------------------
    // Get normalized depth at specific pixel
    //------------------------------------------------------------------------
    float GetNormalizedDepthAt(int x, int y)
    {
        if (x < 0 || x >= m_iGridWidth || y < 0 || y >= m_iGridHeight)
            return 1.0;

        int index = y * m_iGridWidth + x;
        return m_aNormalizedDepth[index];
    }

    //------------------------------------------------------------------------
    // Check if depth is valid at pixel
    //------------------------------------------------------------------------
    bool IsValidAt(int x, int y)
    {
        if (x < 0 || x >= m_iGridWidth || y < 0 || y >= m_iGridHeight)
            return false;

        int index = y * m_iGridWidth + x;
        return m_aValidityMask[index];
    }

    //------------------------------------------------------------------------
    // Get grid dimensions
    //------------------------------------------------------------------------
    int GetGridWidth()
    {
        return m_iGridWidth;
    }

    int GetGridHeight()
    {
        return m_iGridHeight;
    }

    int GetTotalPixels()
    {
        return m_iTotalPixels;
    }

    //------------------------------------------------------------------------
    // Get depth statistics
    //------------------------------------------------------------------------
    float GetMinDepth()
    {
        return m_fMinDepth;
    }

    float GetMaxDepth()
    {
        return m_fMaxDepth;
    }

    float GetMeanDepth()
    {
        return m_fMeanDepth;
    }

    //------------------------------------------------------------------------
    // Get performance statistics
    //------------------------------------------------------------------------
    float GetLastCaptureTimeMs()
    {
        return m_fLastCaptureTimeMs;
    }

    float GetAverageCaptureTimeMs()
    {
        return m_fAverageCaptureTimeMs;
    }

    int GetRaysCast()
    {
        return m_iRaysCast;
    }

    int GetRaysHit()
    {
        return m_iRaysHit;
    }

    float GetHitRate()
    {
        if (m_iRaysCast == 0)
            return 0;
        return m_iRaysHit / m_iRaysCast;
    }

    //------------------------------------------------------------------------
    // Export depth as flat float array for ML pipeline
    // Format: row-major, top-to-bottom, left-to-right
    //------------------------------------------------------------------------
    void ExportAsFloatArray(out array<float> outDepths, out array<float> outNormalized,
                            out int outWidth, out int outHeight)
    {
        outWidth = m_iGridWidth;
        outHeight = m_iGridHeight;

        outDepths = new array<float>();
        outNormalized = new array<float>();

        for (int i = 0; i < m_iTotalPixels; i++)
        {
            outDepths.Insert(m_aDepthBuffer[i]);
            outNormalized.Insert(m_aNormalizedDepth[i]);
        }
    }

    //------------------------------------------------------------------------
    // Export depth map to CSV file (for debugging/analysis)
    //------------------------------------------------------------------------
    bool ExportToCSV(string filePath, bool normalized = true)
    {
        FileHandle file = FileIO.OpenFile(filePath, FileMode.WRITE);
        if (!file)
        {
            Print("[DepthRaycaster] ERROR: Cannot open file: " + filePath, LogLevel.ERROR);
            return false;
        }

        // Write header
        file.WriteLine("# Depth Map Export");
        file.WriteLine("# Width: " + m_iGridWidth.ToString());
        file.WriteLine("# Height: " + m_iGridHeight.ToString());
        file.WriteLine("# FOV: " + m_fVerticalFOV.ToString(5, 2));
        file.WriteLine("# Near: " + m_fNearPlane.ToString(6, 2));
        file.WriteLine("# Far: " + m_fFarPlane.ToString(6, 2));
        file.WriteLine("# MinDepth: " + m_fMinDepth.ToString(8, 3));
        file.WriteLine("# MaxDepth: " + m_fMaxDepth.ToString(8, 3));
        file.WriteLine("# MeanDepth: " + m_fMeanDepth.ToString(8, 3));

        // Write data as comma-separated rows
        for (int y = 0; y < m_iGridHeight; y++)
        {
            string row = "";
            for (int x = 0; x < m_iGridWidth; x++)
            {
                int index = y * m_iGridWidth + x;
                float value;
                if (normalized)
                    value = m_aNormalizedDepth[index];
                else
                    value = m_aDepthBuffer[index];

                if (x > 0)
                    row += ",";
                row += value.ToString(8, 4);
            }
            file.WriteLine(row);
        }

        file.Close();

        Print("[DepthRaycaster] Exported depth map to: " + filePath, LogLevel.NORMAL);
        return true;
    }

    //------------------------------------------------------------------------
    // Export depth map in binary format (more efficient for ML)
    // Format: [width:4][height:4][data:width*height*4] (all floats)
    //------------------------------------------------------------------------
    bool ExportToBinary(string filePath, bool normalized = true)
    {
        FileHandle file = FileIO.OpenFile(filePath, FileMode.WRITE);
        if (!file)
        {
            Print("[DepthRaycaster] ERROR: Cannot open file: " + filePath, LogLevel.ERROR);
            return false;
        }

        // Write header (dimensions as text for compatibility)
        file.WriteLine(m_iGridWidth.ToString() + " " + m_iGridHeight.ToString());

        // Write depth values
        for (int i = 0; i < m_iTotalPixels; i++)
        {
            float value;
            if (normalized)
                value = m_aNormalizedDepth[i];
            else
                value = m_aDepthBuffer[i];

            // Write as text (Enforce script doesn't have binary write for floats)
            file.WriteLine(value.ToString(10, 6));
        }

        file.Close();

        Print("[DepthRaycaster] Exported binary depth to: " + filePath, LogLevel.NORMAL);
        return true;
    }

    //------------------------------------------------------------------------
    // Set configuration at runtime
    //------------------------------------------------------------------------
    void SetGridResolution(int width, int height)
    {
        if (width != m_iGridWidth || height != m_iGridHeight)
        {
            Initialize(width, height, m_fVerticalFOV);
        }
    }

    void SetFOV(float verticalFOV)
    {
        if (Math.AbsFloat(verticalFOV - m_fVerticalFOV) > 0.1)
        {
            m_fVerticalFOV = verticalFOV;
            ComputeRayDirections();
        }
    }

    void SetAspectRatio(float aspectRatio)
    {
        if (Math.AbsFloat(aspectRatio - m_fAspectRatio) > 0.01)
        {
            m_fAspectRatio = aspectRatio;
            ComputeRayDirections();
        }
    }

    void SetDepthRange(float nearPlane, float farPlane)
    {
        m_fNearPlane = nearPlane;
        m_fFarPlane = farPlane;
    }

    void SetLayerMask(int layerMask)
    {
        if (m_TraceParam)
            m_TraceParam.LayerMask = layerMask;
    }

    void EnableCheckerboard(bool enable)
    {
        m_bCheckerboardOptimization = enable;
    }

    void EnableTemporalCache(bool enable, int frames = 3)
    {
        m_bEnableTemporalCache = enable;
        m_iTemporalCacheFrames = frames;
        if (enable && m_bInitialized)
        {
            InitializeTemporalCache();
        }
    }

    //------------------------------------------------------------------------
    // Print debug information
    //------------------------------------------------------------------------
    void PrintDebugInfo()
    {
        Print("=== DEPTH RAYCASTER DEBUG ===", LogLevel.NORMAL);
        Print("Resolution: " + m_iGridWidth + "x" + m_iGridHeight + " (" + m_iTotalPixels + " pixels)", LogLevel.NORMAL);
        Print("FOV: " + m_fVerticalFOV.ToString(5, 1) + " deg, Aspect: " + m_fAspectRatio.ToString(4, 3), LogLevel.NORMAL);
        Print("Depth range: " + m_fNearPlane.ToString(4, 2) + " - " + m_fFarPlane.ToString(6, 1) + " m", LogLevel.NORMAL);
        Print("Rays cast: " + m_iRaysCast + ", Hit: " + m_iRaysHit + " (" + (GetHitRate() * 100).ToString(4, 1) + "%)", LogLevel.NORMAL);
        Print("Depth stats: min=" + m_fMinDepth.ToString(6, 2) + " max=" + m_fMaxDepth.ToString(6, 2) + " mean=" + m_fMeanDepth.ToString(6, 2), LogLevel.NORMAL);
        Print("Capture time: " + m_fLastCaptureTimeMs.ToString(6, 2) + " ms (avg: " + m_fAverageCaptureTimeMs.ToString(6, 2) + " ms)", LogLevel.NORMAL);
        Print("Captures: " + m_iCaptureCount, LogLevel.NORMAL);
        Print("===============================", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Check if initialized
    //------------------------------------------------------------------------
    bool IsInitialized()
    {
        return m_bInitialized;
    }
}

// ============================================================================
// SCR_DepthRaycasterManager - Singleton manager for depth capture
// ============================================================================
class SCR_DepthRaycasterManager
{
    protected static ref SCR_DepthRaycasterManager s_Instance;
    protected ref SCR_DepthRaycaster m_Raycaster;

    //------------------------------------------------------------------------
    static SCR_DepthRaycasterManager GetInstance()
    {
        if (!s_Instance)
        {
            s_Instance = new SCR_DepthRaycasterManager();
        }
        return s_Instance;
    }

    //------------------------------------------------------------------------
    void Initialize(int width = 128, int height = 96, float fov = 75.0)
    {
        if (!m_Raycaster)
        {
            m_Raycaster = new SCR_DepthRaycaster();
        }
        m_Raycaster.Initialize(width, height, fov);
    }

    //------------------------------------------------------------------------
    SCR_DepthRaycaster GetRaycaster()
    {
        return m_Raycaster;
    }

    //------------------------------------------------------------------------
    bool CaptureCurrentFrame()
    {
        if (!m_Raycaster)
            return false;
        return m_Raycaster.CaptureFromCurrentCamera();
    }

    //------------------------------------------------------------------------
    array<float> GetDepthBuffer()
    {
        if (!m_Raycaster)
            return null;
        return m_Raycaster.GetDepthBuffer();
    }

    //------------------------------------------------------------------------
    array<float> GetNormalizedDepthBuffer()
    {
        if (!m_Raycaster)
            return null;
        return m_Raycaster.GetNormalizedDepthBuffer();
    }
}
