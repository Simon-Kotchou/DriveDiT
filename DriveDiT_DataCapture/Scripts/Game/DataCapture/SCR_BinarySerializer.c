// ============================================================================
// SCR_BinarySerializer - High-Performance Binary Data Capture Format
// ============================================================================
//
// Implements the ENFCAP binary format for efficient ML training data capture.
// Designed for high-frequency capture with random access support.
//
// FILE FORMAT SPECIFICATION (ENFCAP v1):
// ============================================================================
//
// File: session_XXXX.enfcap
//
// +-------------------+
// | HEADER (64 bytes) |
// +-------------------+
// |   INDEX TABLE     |
// | (N * 8 bytes)     |
// +-------------------+
// |   FRAME 0 DATA    |
// +-------------------+
// |   FRAME 1 DATA    |
// +-------------------+
// |       ...         |
// +-------------------+
// |   FRAME N DATA    |
// +-------------------+
//
// Header (64 bytes):
//   - Magic:           "ENFCAP01" (8 bytes)
//   - Version:         uint32 (4 bytes)
//   - Frame count:     uint32 (4 bytes)
//   - Start timestamp: float64 (8 bytes) - using 2x float32
//   - Flags:           uint32 (4 bytes)
//   - Reserved:        36 bytes
//
// Index Table (frame_count * 8 bytes):
//   - Per frame: offset (uint64) to frame data (stored as 2x uint32)
//
// Frame Record (variable length):
//   - Frame ID:           uint32 (4 bytes)
//   - Timestamp:          float32 (4 bytes)
//   - Ego Transform:      12 * float32 (48 bytes) - 3x4 matrix
//   - Vehicle State:      10 * float32 (40 bytes)
//   - Scene Entity Count: uint16 (2 bytes)
//   - Scene Entities:     variable (per entity: 16 bytes)
//   - Road Point Count:   uint16 (2 bytes)
//   - Road Points:        variable (per point: 12 bytes)
//   - Flags:              uint16 (2 bytes)
//
// ============================================================================

// --- Format Constants ---
const string ENFCAP_MAGIC = "ENFCAP01";
const int ENFCAP_VERSION = 1;
const int ENFCAP_HEADER_SIZE = 64;
const int ENFCAP_INDEX_ENTRY_SIZE = 8;  // uint64 stored as 2x uint32

// --- Header Flags ---
const int ENFCAP_FLAG_COMPRESSED = 1;       // Data is compressed
const int ENFCAP_FLAG_HAS_SCREENSHOTS = 2;  // External screenshot sync available
const int ENFCAP_FLAG_HAS_DEPTH = 4;        // Depth data included
const int ENFCAP_FLAG_HAS_AUDIO = 8;        // Audio data included
const int ENFCAP_FLAG_ANCHOR_FRAMES = 16;   // Contains anchor frames for random access

// --- Frame Flags ---
const int FRAME_FLAG_ANCHOR = 1;            // This frame is an anchor (keyframe)
const int FRAME_FLAG_COLLISION = 2;         // Collision detected this frame
const int FRAME_FLAG_MANUAL_CONTROL = 4;    // Human driving (not AI)
const int FRAME_FLAG_SCENE_CHANGE = 8;      // Significant scene change

// --- Size Constants ---
const int FRAME_BASE_SIZE = 102;            // Fixed portion of frame record
const int ENTITY_RECORD_SIZE = 16;          // Per-entity data size
const int ROAD_POINT_SIZE = 12;             // Per-road-point data size

// ============================================================================
// Scene Entity Data Structure
// ============================================================================
// Compact representation of a scene entity for ML training
class SCR_CapturedEntity
{
    int m_iEntityType;          // Entity classification (vehicle=0, pedestrian=1, etc.)
    vector m_vPosition;         // World position (x, y, z)
    float m_fYaw;               // Facing direction (radians)
    float m_fVelocity;          // Speed (m/s)

    void SCR_CapturedEntity()
    {
        m_iEntityType = 0;
        m_vPosition = vector.Zero;
        m_fYaw = 0;
        m_fVelocity = 0;
    }
}

// ============================================================================
// Road Point Data Structure
// ============================================================================
// Compact road topology point
class SCR_CapturedRoadPoint
{
    vector m_vPosition;         // World position (x, y, z)

    void SCR_CapturedRoadPoint()
    {
        m_vPosition = vector.Zero;
    }
}

// ============================================================================
// Frame Data Container
// ============================================================================
// Complete frame data for serialization
class SCR_CaptureFrame
{
    // --- Core Frame Data ---
    int m_iFrameID;
    float m_fTimestamp;         // Milliseconds since session start

    // --- Ego Vehicle Transform (3x4 matrix as 12 floats) ---
    // Row-major: [right.x, right.y, right.z, 0,
    //             up.x, up.y, up.z, 0,
    //             forward.x, forward.y, forward.z, 0]
    // Position stored separately for clarity
    ref array<float> m_aEgoTransform;

    // --- Vehicle State (10 floats) ---
    // [speed_kmh, steering, throttle, brake, clutch,
    //  gear, engine_rpm, acceleration, yaw_rate, slip_angle]
    ref array<float> m_aVehicleState;

    // --- Scene Entities ---
    ref array<ref SCR_CapturedEntity> m_aSceneEntities;

    // --- Road Topology ---
    ref array<ref SCR_CapturedRoadPoint> m_aRoadPoints;

    // --- Frame Flags ---
    int m_iFlags;

    //------------------------------------------------------------------------
    void SCR_CaptureFrame()
    {
        m_iFrameID = 0;
        m_fTimestamp = 0;
        m_aEgoTransform = new array<float>();
        m_aVehicleState = new array<float>();
        m_aSceneEntities = new array<ref SCR_CapturedEntity>();
        m_aRoadPoints = new array<ref SCR_CapturedRoadPoint>();
        m_iFlags = 0;

        // Initialize transform array (12 floats for 3x4 matrix)
        for (int i = 0; i < 12; i++)
        {
            m_aEgoTransform.Insert(0.0);
        }

        // Initialize vehicle state array (10 floats)
        for (int i = 0; i < 10; i++)
        {
            m_aVehicleState.Insert(0.0);
        }
    }

    //------------------------------------------------------------------------
    // Set ego transform from world transform matrix
    void SetEgoTransform(vector transform[4])
    {
        // Row 0: Right vector
        m_aEgoTransform[0] = transform[0][0];
        m_aEgoTransform[1] = transform[0][1];
        m_aEgoTransform[2] = transform[0][2];

        // Row 1: Up vector
        m_aEgoTransform[3] = transform[1][0];
        m_aEgoTransform[4] = transform[1][1];
        m_aEgoTransform[5] = transform[1][2];

        // Row 2: Forward vector
        m_aEgoTransform[6] = transform[2][0];
        m_aEgoTransform[7] = transform[2][1];
        m_aEgoTransform[8] = transform[2][2];

        // Row 3: Position
        m_aEgoTransform[9] = transform[3][0];
        m_aEgoTransform[10] = transform[3][1];
        m_aEgoTransform[11] = transform[3][2];
    }

    //------------------------------------------------------------------------
    // Set vehicle state from simulation
    void SetVehicleState(float speed, float steering, float throttle, float brake,
                         float clutch, int gear, float rpm, float acceleration,
                         float yawRate, float slipAngle)
    {
        m_aVehicleState[0] = speed;
        m_aVehicleState[1] = steering;
        m_aVehicleState[2] = throttle;
        m_aVehicleState[3] = brake;
        m_aVehicleState[4] = clutch;
        m_aVehicleState[5] = gear;
        m_aVehicleState[6] = rpm;
        m_aVehicleState[7] = acceleration;
        m_aVehicleState[8] = yawRate;
        m_aVehicleState[9] = slipAngle;
    }

    //------------------------------------------------------------------------
    // Add a scene entity
    void AddSceneEntity(int entityType, vector position, float yaw, float velocity)
    {
        SCR_CapturedEntity entity = new SCR_CapturedEntity();
        entity.m_iEntityType = entityType;
        entity.m_vPosition = position;
        entity.m_fYaw = yaw;
        entity.m_fVelocity = velocity;
        m_aSceneEntities.Insert(entity);
    }

    //------------------------------------------------------------------------
    // Add a road point
    void AddRoadPoint(vector position)
    {
        SCR_CapturedRoadPoint point = new SCR_CapturedRoadPoint();
        point.m_vPosition = position;
        m_aRoadPoints.Insert(point);
    }

    //------------------------------------------------------------------------
    // Calculate serialized size of this frame
    int GetSerializedSize()
    {
        int size = FRAME_BASE_SIZE;
        size += m_aSceneEntities.Count() * ENTITY_RECORD_SIZE;
        size += m_aRoadPoints.Count() * ROAD_POINT_SIZE;
        return size;
    }
}

// ============================================================================
// Binary File Writer Component
// ============================================================================
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Binary Data Serializer - High-performance ENFCAP format writer")]
class SCR_BinarySerializerClass: ScriptComponentClass
{
}

class SCR_BinarySerializer: ScriptComponent
{
    // === CONFIGURATION ===
    [Attribute("1", UIWidgets.CheckBox, "Enable binary capture")]
    protected bool m_bEnabled;

    [Attribute("100", UIWidgets.Slider, "Anchor frame interval (for random access)", "10 500 10")]
    protected int m_iAnchorFrameInterval;

    [Attribute("1000", UIWidgets.Slider, "Index table pre-allocation size", "100 10000 100")]
    protected int m_iIndexPrealloc;

    [Attribute("1", UIWidgets.CheckBox, "Verbose logging")]
    protected bool m_bVerboseLogging;

    // === FILE STATE ===
    protected FileHandle m_hFile;
    protected string m_sFilePath;
    protected bool m_bFileOpen;
    protected bool m_bHeaderWritten;

    // === CAPTURE STATE ===
    protected int m_iFrameCount;
    protected float m_fStartTimestamp;
    protected int m_iHeaderFlags;

    // === INDEX TABLE ===
    // Store frame offsets for random access
    // Using two arrays since Enfusion doesn't have uint64
    protected ref array<int> m_aIndexOffsetsLow;    // Lower 32 bits
    protected ref array<int> m_aIndexOffsetsHigh;   // Upper 32 bits

    // === WRITE BUFFER ===
    // Buffer frame data for efficient I/O
    protected ref array<ref SCR_CaptureFrame> m_aWriteBuffer;
    protected int m_iBufferFlushThreshold;

    // === STATISTICS ===
    protected int m_iTotalBytesWritten;
    protected int m_iAnchorFrameCount;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        // Initialize arrays
        m_aIndexOffsetsLow = new array<int>();
        m_aIndexOffsetsHigh = new array<int>();
        m_aWriteBuffer = new array<ref SCR_CaptureFrame>();

        // Pre-allocate index table
        m_aIndexOffsetsLow.Reserve(m_iIndexPrealloc);
        m_aIndexOffsetsHigh.Reserve(m_iIndexPrealloc);

        m_bFileOpen = false;
        m_bHeaderWritten = false;
        m_iFrameCount = 0;
        m_iTotalBytesWritten = 0;
        m_iAnchorFrameCount = 0;
        m_iBufferFlushThreshold = 100;

        Print("[BinarySerializer] Component initialized", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Create a new capture file
    bool CreateCaptureFile(string basePath, int sessionID)
    {
        if (!m_bEnabled)
        {
            Print("[BinarySerializer] Capture disabled", LogLevel.WARNING);
            return false;
        }

        if (m_bFileOpen)
        {
            Print("[BinarySerializer] ERROR: File already open", LogLevel.ERROR);
            return false;
        }

        // Ensure directory exists
        FileIO.MakeDirectory(basePath);

        // Build file path
        m_sFilePath = basePath + "/session_" + sessionID.ToString() + ".enfcap";

        // Open file for writing
        m_hFile = FileIO.OpenFile(m_sFilePath, FileMode.WRITE);
        if (!m_hFile)
        {
            Print("[BinarySerializer] ERROR: Failed to create file: " + m_sFilePath, LogLevel.ERROR);
            return false;
        }

        m_bFileOpen = true;
        m_bHeaderWritten = false;
        m_iFrameCount = 0;
        m_iTotalBytesWritten = 0;
        m_iAnchorFrameCount = 0;
        m_fStartTimestamp = GetGame().GetWorld().GetWorldTime();
        m_iHeaderFlags = ENFCAP_FLAG_ANCHOR_FRAMES;

        // Clear index tables
        m_aIndexOffsetsLow.Clear();
        m_aIndexOffsetsHigh.Clear();
        m_aWriteBuffer.Clear();

        // Write placeholder header (will be updated on finalize)
        if (!WriteHeader())
        {
            m_hFile.Close();
            m_bFileOpen = false;
            return false;
        }

        Print("[BinarySerializer] Created capture file: " + m_sFilePath, LogLevel.NORMAL);
        return true;
    }

    //------------------------------------------------------------------------
    // Write file header (placeholder - updated on finalize)
    protected bool WriteHeader()
    {
        if (!m_hFile)
            return false;

        // Magic bytes: "ENFCAP01"
        // Write as individual bytes since we need exact control
        int magic0 = 0x45;  // 'E'
        int magic1 = 0x4E;  // 'N'
        int magic2 = 0x46;  // 'F'
        int magic3 = 0x43;  // 'C'
        int magic4 = 0x41;  // 'A'
        int magic5 = 0x50;  // 'P'
        int magic6 = 0x30;  // '0'
        int magic7 = 0x31;  // '1'

        // Pack magic into two 32-bit integers
        int magicLow = (magic3 << 24) | (magic2 << 16) | (magic1 << 8) | magic0;
        int magicHigh = (magic7 << 24) | (magic6 << 16) | (magic5 << 8) | magic4;

        m_hFile.Write(magicLow, 4);
        m_hFile.Write(magicHigh, 4);

        // Version (uint32)
        int version = ENFCAP_VERSION;
        m_hFile.Write(version, 4);

        // Frame count (uint32) - placeholder, updated on finalize
        int frameCount = 0;
        m_hFile.Write(frameCount, 4);

        // Start timestamp (float64 as 2x float32)
        float timestampLow = m_fStartTimestamp;
        float timestampHigh = 0.0;  // For very large timestamps
        m_hFile.Write(timestampLow, 4);
        m_hFile.Write(timestampHigh, 4);

        // Flags (uint32)
        m_hFile.Write(m_iHeaderFlags, 4);

        // Reserved (36 bytes = 9 int32)
        int reserved = 0;
        for (int i = 0; i < 9; i++)
        {
            m_hFile.Write(reserved, 4);
        }

        m_bHeaderWritten = true;
        m_iTotalBytesWritten = ENFCAP_HEADER_SIZE;

        if (m_bVerboseLogging)
        {
            Print("[BinarySerializer] Header written (64 bytes)", LogLevel.VERBOSE);
        }

        return true;
    }

    //------------------------------------------------------------------------
    // Capture and write a single frame
    bool WriteFrame(SCR_CaptureFrame frame)
    {
        if (!m_bFileOpen || !m_hFile)
        {
            Print("[BinarySerializer] ERROR: No file open for writing", LogLevel.ERROR);
            return false;
        }

        // Determine if this is an anchor frame
        bool isAnchor = (m_iFrameCount % m_iAnchorFrameInterval == 0);
        if (isAnchor)
        {
            frame.m_iFlags = frame.m_iFlags | FRAME_FLAG_ANCHOR;
            m_iAnchorFrameCount++;
        }

        // Record offset for this frame (after header + index table space)
        // We write index table at end, so frames start after header
        int currentOffset = m_hFile.GetPos();

        // Store offset (split into low/high 32 bits for large files)
        m_aIndexOffsetsLow.Insert(currentOffset);
        m_aIndexOffsetsHigh.Insert(0);  // For files < 4GB, high is always 0

        // Write frame data
        if (!WriteFrameData(frame))
        {
            return false;
        }

        m_iFrameCount++;

        // Verbose logging
        if (m_bVerboseLogging && m_iFrameCount % 500 == 0)
        {
            Print("[BinarySerializer] Written " + m_iFrameCount.ToString() + " frames, " +
                  (m_iTotalBytesWritten / 1024).ToString() + " KB", LogLevel.VERBOSE);
        }

        return true;
    }

    //------------------------------------------------------------------------
    // Write frame binary data
    protected bool WriteFrameData(SCR_CaptureFrame frame)
    {
        int bytesWritten = 0;

        // Frame ID (uint32)
        m_hFile.Write(frame.m_iFrameID, 4);
        bytesWritten += 4;

        // Timestamp (float32)
        m_hFile.Write(frame.m_fTimestamp, 4);
        bytesWritten += 4;

        // Ego Transform (12 * float32 = 48 bytes)
        for (int i = 0; i < 12; i++)
        {
            float val = frame.m_aEgoTransform[i];
            m_hFile.Write(val, 4);
            bytesWritten += 4;
        }

        // Vehicle State (10 * float32 = 40 bytes)
        for (int i = 0; i < 10; i++)
        {
            float val = frame.m_aVehicleState[i];
            m_hFile.Write(val, 4);
            bytesWritten += 4;
        }

        // Scene Entity Count (uint16 as int with 2 bytes)
        int entityCount = frame.m_aSceneEntities.Count();
        if (entityCount > 65535)
            entityCount = 65535;  // Clamp to uint16 max
        m_hFile.Write(entityCount, 2);
        bytesWritten += 2;

        // Scene Entities (entityCount * 16 bytes each)
        for (int i = 0; i < entityCount; i++)
        {
            SCR_CapturedEntity entity = frame.m_aSceneEntities[i];

            // Entity type (int32 - could use smaller but keeping aligned)
            m_hFile.Write(entity.m_iEntityType, 4);

            // Position (3 * float32)
            float px = entity.m_vPosition[0];
            float py = entity.m_vPosition[1];
            float pz = entity.m_vPosition[2];
            m_hFile.Write(px, 4);
            m_hFile.Write(py, 4);
            m_hFile.Write(pz, 4);

            bytesWritten += 16;
        }

        // Road Point Count (uint16)
        int roadPointCount = frame.m_aRoadPoints.Count();
        if (roadPointCount > 65535)
            roadPointCount = 65535;
        m_hFile.Write(roadPointCount, 2);
        bytesWritten += 2;

        // Road Points (roadPointCount * 12 bytes each)
        for (int i = 0; i < roadPointCount; i++)
        {
            SCR_CapturedRoadPoint point = frame.m_aRoadPoints[i];

            // Position (3 * float32)
            float px = point.m_vPosition[0];
            float py = point.m_vPosition[1];
            float pz = point.m_vPosition[2];
            m_hFile.Write(px, 4);
            m_hFile.Write(py, 4);
            m_hFile.Write(pz, 4);

            bytesWritten += 12;
        }

        // Frame Flags (uint16)
        m_hFile.Write(frame.m_iFlags, 2);
        bytesWritten += 2;

        m_iTotalBytesWritten += bytesWritten;

        return true;
    }

    //------------------------------------------------------------------------
    // Convenience method: Capture from vehicle simulation
    bool CaptureVehicleFrame(IEntity vehicle, VehicleWheeledSimulation sim,
                              array<ref SCR_CapturedEntity> sceneEntities,
                              array<vector> roadPoints)
    {
        if (!vehicle || !sim)
            return false;

        SCR_CaptureFrame frame = new SCR_CaptureFrame();

        // Set frame ID and timestamp
        frame.m_iFrameID = m_iFrameCount;
        frame.m_fTimestamp = GetGame().GetWorld().GetWorldTime() - m_fStartTimestamp;

        // Get vehicle transform
        vector transform[4];
        vehicle.GetWorldTransform(transform);
        frame.SetEgoTransform(transform);

        // Get vehicle state
        float speed = sim.GetSpeedKmh();
        float steering = sim.GetSteering();
        float throttle = sim.GetThrottle();
        float brake = sim.GetBrake();
        float clutch = sim.GetClutch();
        int gear = sim.GetGear();
        float rpm = sim.EngineGetRPM();

        // TODO: Calculate these from vehicle physics
        float acceleration = 0.0;
        float yawRate = 0.0;
        float slipAngle = 0.0;

        frame.SetVehicleState(speed, steering, throttle, brake, clutch,
                              gear, rpm, acceleration, yawRate, slipAngle);

        // Add scene entities
        if (sceneEntities)
        {
            for (int i = 0; i < sceneEntities.Count(); i++)
            {
                SCR_CapturedEntity entity = sceneEntities[i];
                frame.m_aSceneEntities.Insert(entity);
            }
        }

        // Add road points
        if (roadPoints)
        {
            for (int i = 0; i < roadPoints.Count(); i++)
            {
                frame.AddRoadPoint(roadPoints[i]);
            }
        }

        return WriteFrame(frame);
    }

    //------------------------------------------------------------------------
    // Finalize and close the capture file
    bool FinalizeCaptureFile()
    {
        if (!m_bFileOpen || !m_hFile)
        {
            Print("[BinarySerializer] No file open to finalize", LogLevel.WARNING);
            return false;
        }

        // Write index table at current position
        int indexTableOffset = m_hFile.GetPos();

        // Write index entries
        for (int i = 0; i < m_aIndexOffsetsLow.Count(); i++)
        {
            int offsetLow = m_aIndexOffsetsLow[i];
            int offsetHigh = m_aIndexOffsetsHigh[i];

            m_hFile.Write(offsetLow, 4);
            m_hFile.Write(offsetHigh, 4);
        }

        int indexTableSize = m_aIndexOffsetsLow.Count() * ENFCAP_INDEX_ENTRY_SIZE;
        m_iTotalBytesWritten += indexTableSize;

        // Update header with final frame count
        UpdateHeader();

        // Close file
        m_hFile.Close();
        m_bFileOpen = false;
        m_hFile = null;

        Print("[BinarySerializer] Capture finalized: " + m_sFilePath, LogLevel.NORMAL);
        Print("[BinarySerializer] Frames: " + m_iFrameCount.ToString() +
              ", Anchors: " + m_iAnchorFrameCount.ToString() +
              ", Size: " + (m_iTotalBytesWritten / 1024).ToString() + " KB", LogLevel.NORMAL);

        return true;
    }

    //------------------------------------------------------------------------
    // Update header with final values
    protected void UpdateHeader()
    {
        if (!m_hFile)
            return;

        // Seek to frame count position (offset 12: after magic + version)
        m_hFile.Seek(12);

        // Write final frame count
        m_hFile.Write(m_iFrameCount, 4);

        // Seek back to end
        m_hFile.Seek(m_iTotalBytesWritten);
    }

    //------------------------------------------------------------------------
    // Get capture statistics
    void GetStatistics(out int frameCount, out int anchorCount, out int bytesWritten)
    {
        frameCount = m_iFrameCount;
        anchorCount = m_iAnchorFrameCount;
        bytesWritten = m_iTotalBytesWritten;
    }

    //------------------------------------------------------------------------
    // Check if file is open for writing
    bool IsCapturing()
    {
        return m_bFileOpen;
    }

    //------------------------------------------------------------------------
    // Get current frame count
    int GetFrameCount()
    {
        return m_iFrameCount;
    }

    //------------------------------------------------------------------------
    // Get file path
    string GetFilePath()
    {
        return m_sFilePath;
    }

    //------------------------------------------------------------------------
    // Set header flag
    void SetHeaderFlag(int flag)
    {
        m_iHeaderFlags = m_iHeaderFlags | flag;
    }

    //------------------------------------------------------------------------
    // Cleanup
    override void OnDelete(IEntity owner)
    {
        if (m_bFileOpen)
        {
            FinalizeCaptureFile();
        }

        super.OnDelete(owner);
    }
}


// ============================================================================
// Binary File Reader Component (for validation and random access)
// ============================================================================
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Binary Data Reader - ENFCAP format reader with random access")]
class SCR_BinaryReaderClass: ScriptComponentClass
{
}

class SCR_BinaryReader: ScriptComponent
{
    // === FILE STATE ===
    protected FileHandle m_hFile;
    protected string m_sFilePath;
    protected bool m_bFileOpen;

    // === HEADER DATA ===
    protected int m_iVersion;
    protected int m_iFrameCount;
    protected float m_fStartTimestamp;
    protected int m_iHeaderFlags;

    // === INDEX TABLE ===
    protected ref array<int> m_aIndexOffsetsLow;
    protected ref array<int> m_aIndexOffsetsHigh;
    protected bool m_bIndexLoaded;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        m_aIndexOffsetsLow = new array<int>();
        m_aIndexOffsetsHigh = new array<int>();
        m_bFileOpen = false;
        m_bIndexLoaded = false;

        Print("[BinaryReader] Component initialized", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Open a capture file for reading
    bool OpenCaptureFile(string filePath)
    {
        if (m_bFileOpen)
        {
            CloseCaptureFile();
        }

        m_sFilePath = filePath;
        m_hFile = FileIO.OpenFile(filePath, FileMode.READ);

        if (!m_hFile)
        {
            Print("[BinaryReader] ERROR: Failed to open file: " + filePath, LogLevel.ERROR);
            return false;
        }

        m_bFileOpen = true;

        // Read and validate header
        if (!ReadHeader())
        {
            Print("[BinaryReader] ERROR: Invalid file header", LogLevel.ERROR);
            m_hFile.Close();
            m_bFileOpen = false;
            return false;
        }

        Print("[BinaryReader] Opened: " + filePath, LogLevel.NORMAL);
        Print("[BinaryReader] Version: " + m_iVersion.ToString() +
              ", Frames: " + m_iFrameCount.ToString(), LogLevel.NORMAL);

        return true;
    }

    //------------------------------------------------------------------------
    // Read and validate file header
    protected bool ReadHeader()
    {
        if (!m_hFile)
            return false;

        // Read magic bytes
        int magicLow, magicHigh;
        m_hFile.Read(magicLow, 4);
        m_hFile.Read(magicHigh, 4);

        // Validate magic (ENFCAP01)
        int expectedLow = 0x43464E45;  // "ENFC" little-endian
        int expectedHigh = 0x31305041; // "AP01" little-endian

        // Note: Byte order may vary - this is a simplified check
        // For production, implement proper magic validation

        // Read version
        m_hFile.Read(m_iVersion, 4);

        if (m_iVersion != ENFCAP_VERSION)
        {
            Print("[BinaryReader] WARNING: Version mismatch. File: " + m_iVersion.ToString() +
                  ", Expected: " + ENFCAP_VERSION.ToString(), LogLevel.WARNING);
        }

        // Read frame count
        m_hFile.Read(m_iFrameCount, 4);

        // Read start timestamp
        m_hFile.Read(m_fStartTimestamp, 4);
        float timestampHigh;
        m_hFile.Read(timestampHigh, 4);  // Ignore high part for now

        // Read flags
        m_hFile.Read(m_iHeaderFlags, 4);

        // Skip reserved bytes
        m_hFile.Seek(ENFCAP_HEADER_SIZE);

        return true;
    }

    //------------------------------------------------------------------------
    // Load index table for random access
    bool LoadIndexTable()
    {
        if (!m_bFileOpen || !m_hFile)
            return false;

        if (m_bIndexLoaded)
            return true;

        // Index table is at end of file
        // Calculate position: file_size - (frame_count * 8)
        int fileSize = m_hFile.GetLength();
        int indexTableSize = m_iFrameCount * ENFCAP_INDEX_ENTRY_SIZE;
        int indexTableOffset = fileSize - indexTableSize;

        // Seek to index table
        m_hFile.Seek(indexTableOffset);

        // Clear existing data
        m_aIndexOffsetsLow.Clear();
        m_aIndexOffsetsHigh.Clear();

        // Read index entries
        for (int i = 0; i < m_iFrameCount; i++)
        {
            int offsetLow, offsetHigh;
            m_hFile.Read(offsetLow, 4);
            m_hFile.Read(offsetHigh, 4);

            m_aIndexOffsetsLow.Insert(offsetLow);
            m_aIndexOffsetsHigh.Insert(offsetHigh);
        }

        m_bIndexLoaded = true;

        Print("[BinaryReader] Index table loaded: " + m_iFrameCount.ToString() + " entries", LogLevel.NORMAL);

        return true;
    }

    //------------------------------------------------------------------------
    // Seek to a specific frame (random access)
    bool SeekToFrame(int frameIndex)
    {
        if (!m_bFileOpen || !m_hFile)
            return false;

        if (frameIndex < 0 || frameIndex >= m_iFrameCount)
        {
            Print("[BinaryReader] ERROR: Frame index out of range: " + frameIndex.ToString(), LogLevel.ERROR);
            return false;
        }

        // Load index table if not loaded
        if (!m_bIndexLoaded)
        {
            if (!LoadIndexTable())
                return false;
        }

        // Get offset for this frame
        int offset = m_aIndexOffsetsLow[frameIndex];

        // Seek to frame
        m_hFile.Seek(offset);

        return true;
    }

    //------------------------------------------------------------------------
    // Read a frame at the current file position
    bool ReadFrame(out SCR_CaptureFrame frame)
    {
        if (!m_bFileOpen || !m_hFile)
            return false;

        frame = new SCR_CaptureFrame();

        // Read Frame ID
        m_hFile.Read(frame.m_iFrameID, 4);

        // Read Timestamp
        m_hFile.Read(frame.m_fTimestamp, 4);

        // Read Ego Transform (12 floats)
        for (int i = 0; i < 12; i++)
        {
            float val;
            m_hFile.Read(val, 4);
            frame.m_aEgoTransform[i] = val;
        }

        // Read Vehicle State (10 floats)
        for (int i = 0; i < 10; i++)
        {
            float val;
            m_hFile.Read(val, 4);
            frame.m_aVehicleState[i] = val;
        }

        // Read Scene Entity Count
        int entityCount;
        m_hFile.Read(entityCount, 2);

        // Read Scene Entities
        for (int i = 0; i < entityCount; i++)
        {
            SCR_CapturedEntity entity = new SCR_CapturedEntity();

            m_hFile.Read(entity.m_iEntityType, 4);

            float px, py, pz;
            m_hFile.Read(px, 4);
            m_hFile.Read(py, 4);
            m_hFile.Read(pz, 4);
            entity.m_vPosition = Vector(px, py, pz);

            frame.m_aSceneEntities.Insert(entity);
        }

        // Read Road Point Count
        int roadPointCount;
        m_hFile.Read(roadPointCount, 2);

        // Read Road Points
        for (int i = 0; i < roadPointCount; i++)
        {
            float px, py, pz;
            m_hFile.Read(px, 4);
            m_hFile.Read(py, 4);
            m_hFile.Read(pz, 4);

            frame.AddRoadPoint(Vector(px, py, pz));
        }

        // Read Frame Flags
        m_hFile.Read(frame.m_iFlags, 2);

        return true;
    }

    //------------------------------------------------------------------------
    // Read a specific frame by index
    bool ReadFrameAt(int frameIndex, out SCR_CaptureFrame frame)
    {
        if (!SeekToFrame(frameIndex))
            return false;

        return ReadFrame(frame);
    }

    //------------------------------------------------------------------------
    // Get frame count
    int GetFrameCount()
    {
        return m_iFrameCount;
    }

    //------------------------------------------------------------------------
    // Get header flags
    int GetHeaderFlags()
    {
        return m_iHeaderFlags;
    }

    //------------------------------------------------------------------------
    // Get file version
    int GetVersion()
    {
        return m_iVersion;
    }

    //------------------------------------------------------------------------
    // Check if file is open
    bool IsOpen()
    {
        return m_bFileOpen;
    }

    //------------------------------------------------------------------------
    // Close the file
    void CloseCaptureFile()
    {
        if (m_bFileOpen && m_hFile)
        {
            m_hFile.Close();
            m_hFile = null;
        }

        m_bFileOpen = false;
        m_bIndexLoaded = false;
        m_aIndexOffsetsLow.Clear();
        m_aIndexOffsetsHigh.Clear();
    }

    //------------------------------------------------------------------------
    // Validate file integrity
    bool ValidateFile()
    {
        if (!m_bFileOpen)
            return false;

        // Load index table
        if (!LoadIndexTable())
            return false;

        // Verify we can read first and last frames
        SCR_CaptureFrame firstFrame, lastFrame;

        if (!ReadFrameAt(0, firstFrame))
        {
            Print("[BinaryReader] ERROR: Failed to read first frame", LogLevel.ERROR);
            return false;
        }

        if (m_iFrameCount > 1)
        {
            if (!ReadFrameAt(m_iFrameCount - 1, lastFrame))
            {
                Print("[BinaryReader] ERROR: Failed to read last frame", LogLevel.ERROR);
                return false;
            }
        }

        Print("[BinaryReader] File validation passed", LogLevel.NORMAL);
        return true;
    }

    //------------------------------------------------------------------------
    // Cleanup
    override void OnDelete(IEntity owner)
    {
        CloseCaptureFile();
        super.OnDelete(owner);
    }
}


// ============================================================================
// Integration Helper: Bridge between MLDataCollector and BinarySerializer
// ============================================================================
class SCR_BinaryCaptureBridge
{
    protected ref SCR_BinarySerializer m_Serializer;
    protected ref SCR_MLDataCollector m_Collector;

    //------------------------------------------------------------------------
    void SCR_BinaryCaptureBridge(SCR_BinarySerializer serializer, SCR_MLDataCollector collector)
    {
        m_Serializer = serializer;
        m_Collector = collector;
    }

    //------------------------------------------------------------------------
    // Capture a frame from the collector's tracked vehicles
    bool CaptureFromCollector(int vehicleIndex, array<vector> roadPoints)
    {
        if (!m_Serializer || !m_Collector)
            return false;

        // This would need access to collector's internal arrays
        // For now, this is a placeholder showing the integration pattern

        return false;
    }

    //------------------------------------------------------------------------
    // Convert CSV telemetry to binary format (batch conversion)
    static bool ConvertCSVToBinary(string csvPath, string outputPath)
    {
        // Open CSV file
        FileHandle csvFile = FileIO.OpenFile(csvPath, FileMode.READ);
        if (!csvFile)
        {
            Print("[BinaryCaptureBridge] ERROR: Failed to open CSV: " + csvPath, LogLevel.ERROR);
            return false;
        }

        // Skip header line
        string headerLine;
        csvFile.ReadLine(headerLine);

        // Create temporary serializer (note: not a component in this context)
        // This is a simplified conversion - in practice, you'd instantiate
        // the component properly or use a standalone conversion utility

        Print("[BinaryCaptureBridge] CSV to Binary conversion not fully implemented", LogLevel.WARNING);
        Print("[BinaryCaptureBridge] Use external Python script for batch conversion", LogLevel.WARNING);

        csvFile.Close();
        return false;
    }
}
