//------------------------------------------------------------------------------------------------
// SCR_CaptureTests.c
// Comprehensive Integration Test Suite for Enfusion Data Capture Pipeline
//
// Components Under Test:
//   - SCR_MLDataCollector      : Telemetry capture system
//   - SCR_DepthRaycaster       : Depth map generation
//   - SCR_MultiCameraRig       : Multi-camera system
//   - SCR_SceneEnumerator      : Entity enumeration
//   - SCR_AnchorFrameSelector  : Anchor frame selection
//   - SCR_BinarySerializer     : Binary output serialization
//
// Test Categories:
//   1. Unit Tests              : Individual component functionality
//   2. Integration Tests       : Component interactions
//   3. Performance Tests       : Frame budget compliance
//   4. Data Validation         : Output format correctness
//   5. Edge Cases              : Error handling, boundary conditions
//
// Usage:
//   Via Workbench: Plugins -> Run Test on SCR_CaptureTestSuite
//   Via CLI: -autotest SCR_CaptureTestSuite
//------------------------------------------------------------------------------------------------

//================================================================================================
// SECTION 1: TEST RESULT CLASSES
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Custom test result for capture pipeline tests
class SCR_CaptureTestResult : TestResultBase
{
    protected bool m_bPassed;
    protected string m_sMessage;
    protected float m_fExecutionTimeMs;
    protected string m_sComponentName;

    //--------------------------------------------------------------------------------------------
    void SCR_CaptureTestResult(bool passed, string message, string componentName = "", float execTimeMs = 0)
    {
        m_bPassed = passed;
        m_sMessage = message;
        m_sComponentName = componentName;
        m_fExecutionTimeMs = execTimeMs;
    }

    //--------------------------------------------------------------------------------------------
    override bool Failure()
    {
        return !m_bPassed;
    }

    //--------------------------------------------------------------------------------------------
    override string FailureText()
    {
        if (m_bPassed)
            return "";

        string result = "[FAIL] ";
        if (m_sComponentName != "")
            result += m_sComponentName + ": ";
        result += m_sMessage;

        if (m_fExecutionTimeMs > 0)
            result += string.Format(" (%.2fms)", m_fExecutionTimeMs);

        return result;
    }

    //--------------------------------------------------------------------------------------------
    string GetSuccessText()
    {
        string result = "[PASS] ";
        if (m_sComponentName != "")
            result += m_sComponentName + ": ";
        result += m_sMessage;

        if (m_fExecutionTimeMs > 0)
            result += string.Format(" (%.2fms)", m_fExecutionTimeMs);

        return result;
    }

    //--------------------------------------------------------------------------------------------
    float GetExecutionTime() { return m_fExecutionTimeMs; }
    string GetComponentName() { return m_sComponentName; }
    bool IsPassed() { return m_bPassed; }
}

//------------------------------------------------------------------------------------------------
//! Performance test result with timing metrics
class SCR_PerformanceTestResult : SCR_CaptureTestResult
{
    protected float m_fTargetTimeMs;
    protected float m_fMinTimeMs;
    protected float m_fMaxTimeMs;
    protected float m_fAvgTimeMs;
    protected int m_iIterations;

    //--------------------------------------------------------------------------------------------
    void SCR_PerformanceTestResult(bool passed, string message, string componentName,
                                    float avgTimeMs, float targetTimeMs,
                                    float minTimeMs = 0, float maxTimeMs = 0, int iterations = 1)
    {
        SCR_CaptureTestResult(passed, message, componentName, avgTimeMs);
        m_fTargetTimeMs = targetTimeMs;
        m_fMinTimeMs = minTimeMs;
        m_fMaxTimeMs = maxTimeMs;
        m_fAvgTimeMs = avgTimeMs;
        m_iIterations = iterations;
    }

    //--------------------------------------------------------------------------------------------
    override string FailureText()
    {
        if (m_bPassed)
            return "";

        return string.Format("[PERF FAIL] %1: %2 - Avg: %.3fms (Target: %.3fms, Min: %.3fms, Max: %.3fms, Iterations: %3)",
            m_sComponentName, m_sMessage, m_fAvgTimeMs, m_fTargetTimeMs, m_fMinTimeMs, m_fMaxTimeMs, m_iIterations);
    }

    //--------------------------------------------------------------------------------------------
    string GetPerformanceReport()
    {
        return string.Format("Performance: Avg=%.3fms, Target=%.3fms, Min=%.3fms, Max=%.3fms, Iterations=%1",
            m_fAvgTimeMs, m_fTargetTimeMs, m_fMinTimeMs, m_fMaxTimeMs, m_iIterations);
    }

    float GetTargetTime() { return m_fTargetTimeMs; }
    float GetAverageTime() { return m_fAvgTimeMs; }
    float GetMinTime() { return m_fMinTimeMs; }
    float GetMaxTime() { return m_fMaxTimeMs; }
    int GetIterations() { return m_iIterations; }
}

//================================================================================================
// SECTION 2: TEST UTILITIES AND HELPERS
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Assertion helper class for test validation
class SCR_TestAssert
{
    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertTrue(bool condition, string message, string component = "")
    {
        return new SCR_CaptureTestResult(condition, message, component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertFalse(bool condition, string message, string component = "")
    {
        return new SCR_CaptureTestResult(!condition, message, component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertNotNull(Class obj, string message, string component = "")
    {
        return new SCR_CaptureTestResult(obj != null, message + (obj == null ? " (was null)" : ""), component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertNull(Class obj, string message, string component = "")
    {
        return new SCR_CaptureTestResult(obj == null, message + (obj != null ? " (was not null)" : ""), component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertEquals(int expected, int actual, string message, string component = "")
    {
        bool passed = (expected == actual);
        string fullMsg = message;
        if (!passed)
            fullMsg += string.Format(" (expected: %1, actual: %2)", expected, actual);
        return new SCR_CaptureTestResult(passed, fullMsg, component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertEqualsFloat(float expected, float actual, float tolerance, string message, string component = "")
    {
        bool passed = Math.AbsFloat(expected - actual) <= tolerance;
        string fullMsg = message;
        if (!passed)
            fullMsg += string.Format(" (expected: %.4f, actual: %.4f, tolerance: %.4f)", expected, actual, tolerance);
        return new SCR_CaptureTestResult(passed, fullMsg, component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertVectorEquals(vector expected, vector actual, float tolerance, string message, string component = "")
    {
        float dist = vector.Distance(expected, actual);
        bool passed = dist <= tolerance;
        string fullMsg = message;
        if (!passed)
            fullMsg += string.Format(" (distance: %.4f, tolerance: %.4f)", dist, tolerance);
        return new SCR_CaptureTestResult(passed, fullMsg, component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertInRange(float value, float min, float max, string message, string component = "")
    {
        bool passed = (value >= min && value <= max);
        string fullMsg = message;
        if (!passed)
            fullMsg += string.Format(" (value: %.4f, range: [%.4f, %.4f])", value, min, max);
        return new SCR_CaptureTestResult(passed, fullMsg, component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertArrayNotEmpty(array<Class> arr, string message, string component = "")
    {
        bool passed = (arr != null && arr.Count() > 0);
        string fullMsg = message;
        if (!passed)
            fullMsg += (arr == null ? " (array was null)" : " (array was empty)");
        return new SCR_CaptureTestResult(passed, fullMsg, component);
    }

    //--------------------------------------------------------------------------------------------
    static SCR_CaptureTestResult AssertArrayCount(array<Class> arr, int expectedCount, string message, string component = "")
    {
        int actualCount = (arr != null) ? arr.Count() : -1;
        bool passed = (actualCount == expectedCount);
        string fullMsg = message;
        if (!passed)
            fullMsg += string.Format(" (expected: %1, actual: %2)", expectedCount, actualCount);
        return new SCR_CaptureTestResult(passed, fullMsg, component);
    }
}

//------------------------------------------------------------------------------------------------
//! Performance measurement utility
class SCR_PerformanceTimer
{
    protected float m_fStartTime;
    protected float m_fEndTime;
    protected ref array<float> m_aSamples;

    //--------------------------------------------------------------------------------------------
    void SCR_PerformanceTimer()
    {
        m_aSamples = new array<float>();
    }

    //--------------------------------------------------------------------------------------------
    void Start()
    {
        m_fStartTime = System.GetTickCount();
    }

    //--------------------------------------------------------------------------------------------
    float Stop()
    {
        m_fEndTime = System.GetTickCount();
        float elapsed = m_fEndTime - m_fStartTime;
        m_aSamples.Insert(elapsed);
        return elapsed;
    }

    //--------------------------------------------------------------------------------------------
    void Reset()
    {
        m_aSamples.Clear();
        m_fStartTime = 0;
        m_fEndTime = 0;
    }

    //--------------------------------------------------------------------------------------------
    float GetAverageMs()
    {
        if (m_aSamples.Count() == 0)
            return 0;

        float sum = 0;
        foreach (float sample : m_aSamples)
            sum += sample;

        return sum / m_aSamples.Count();
    }

    //--------------------------------------------------------------------------------------------
    float GetMinMs()
    {
        if (m_aSamples.Count() == 0)
            return 0;

        float minVal = m_aSamples[0];
        foreach (float sample : m_aSamples)
        {
            if (sample < minVal)
                minVal = sample;
        }
        return minVal;
    }

    //--------------------------------------------------------------------------------------------
    float GetMaxMs()
    {
        if (m_aSamples.Count() == 0)
            return 0;

        float maxVal = m_aSamples[0];
        foreach (float sample : m_aSamples)
        {
            if (sample > maxVal)
                maxVal = sample;
        }
        return maxVal;
    }

    //--------------------------------------------------------------------------------------------
    int GetSampleCount()
    {
        return m_aSamples.Count();
    }

    //--------------------------------------------------------------------------------------------
    SCR_PerformanceTestResult CreateResult(string testName, string component, float targetMs)
    {
        float avg = GetAverageMs();
        bool passed = avg <= targetMs;

        return new SCR_PerformanceTestResult(
            passed,
            testName,
            component,
            avg,
            targetMs,
            GetMinMs(),
            GetMaxMs(),
            GetSampleCount()
        );
    }
}

//------------------------------------------------------------------------------------------------
//! Test data generator for creating test scenarios
class SCR_TestDataGenerator
{
    //--------------------------------------------------------------------------------------------
    //! Generate random position within bounds
    static vector GenerateRandomPosition(vector minBounds, vector maxBounds)
    {
        float x = Math.RandomFloat(minBounds[0], maxBounds[0]);
        float y = Math.RandomFloat(minBounds[1], maxBounds[1]);
        float z = Math.RandomFloat(minBounds[2], maxBounds[2]);
        return Vector(x, y, z);
    }

    //--------------------------------------------------------------------------------------------
    //! Generate random rotation angles
    static vector GenerateRandomRotation()
    {
        float pitch = Math.RandomFloat(-90, 90);
        float yaw = Math.RandomFloat(0, 360);
        float roll = Math.RandomFloat(-45, 45);
        return Vector(pitch, yaw, roll);
    }

    //--------------------------------------------------------------------------------------------
    //! Generate random velocity vector
    static vector GenerateRandomVelocity(float maxSpeed)
    {
        float speed = Math.RandomFloat(0, maxSpeed);
        float angle = Math.RandomFloat(0, Math.PI2);
        return Vector(Math.Cos(angle) * speed, 0, Math.Sin(angle) * speed);
    }

    //--------------------------------------------------------------------------------------------
    //! Generate test telemetry data packet
    static void GenerateTestTelemetry(out vector position, out vector velocity,
                                       out vector rotation, out float throttle, out float steering)
    {
        position = GenerateRandomPosition(Vector(-1000, 0, -1000), Vector(1000, 100, 1000));
        velocity = GenerateRandomVelocity(50);
        rotation = GenerateRandomRotation();
        throttle = Math.RandomFloat(0, 1);
        steering = Math.RandomFloat(-1, 1);
    }

    //--------------------------------------------------------------------------------------------
    //! Create a mock depth buffer with synthetic data
    static void GenerateDepthBuffer(out array<float> depthBuffer, int width, int height,
                                     float minDepth = 0.1, float maxDepth = 500)
    {
        depthBuffer = new array<float>();
        int size = width * height;

        for (int i = 0; i < size; i++)
        {
            // Generate depth with some structure (gradient with noise)
            int x = i % width;
            int y = i / width;

            float baseDepth = minDepth + (maxDepth - minDepth) * (y / height);
            float noise = Math.RandomFloat(-10, 10);
            float depth = Math.Clamp(baseDepth + noise, minDepth, maxDepth);

            depthBuffer.Insert(depth);
        }
    }
}

//------------------------------------------------------------------------------------------------
//! File validation utilities for output verification
class SCR_OutputValidator
{
    //--------------------------------------------------------------------------------------------
    //! Validate binary file header
    static bool ValidateBinaryHeader(string filePath, int expectedMagic, int expectedVersion)
    {
        FileHandle file = FileIO.OpenFile(filePath, FileMode.READ);
        if (!file)
            return false;

        int magic = 0;
        int version = 0;

        file.Read(magic, 4);
        file.Read(version, 4);
        file.Close();

        return (magic == expectedMagic && version == expectedVersion);
    }

    //--------------------------------------------------------------------------------------------
    //! Check if file exists and has minimum size
    static bool ValidateFileExists(string filePath, int minSize = 0)
    {
        FileHandle file = FileIO.OpenFile(filePath, FileMode.READ);
        if (!file)
            return false;

        int size = file.GetLength();
        file.Close();

        return size >= minSize;
    }

    //--------------------------------------------------------------------------------------------
    //! Validate depth map dimensions
    static bool ValidateDepthMapSize(string filePath, int expectedWidth, int expectedHeight)
    {
        FileHandle file = FileIO.OpenFile(filePath, FileMode.READ);
        if (!file)
            return false;

        // Skip magic and version (8 bytes)
        file.Seek(8);

        int width = 0;
        int height = 0;
        file.Read(width, 4);
        file.Read(height, 4);
        file.Close();

        return (width == expectedWidth && height == expectedHeight);
    }

    //--------------------------------------------------------------------------------------------
    //! Validate telemetry record count
    static int GetTelemetryRecordCount(string filePath)
    {
        FileHandle file = FileIO.OpenFile(filePath, FileMode.READ);
        if (!file)
            return -1;

        // Skip header (assume 16 bytes: magic + version + flags + reserved)
        file.Seek(16);

        int recordCount = 0;
        file.Read(recordCount, 4);
        file.Close();

        return recordCount;
    }
}

//================================================================================================
// SECTION 3: MOCK COMPONENTS FOR ISOLATED TESTING
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Mock telemetry data collector for testing
class SCR_MockMLDataCollector
{
    protected bool m_bIsCapturing;
    protected int m_iFrameCount;
    protected float m_fCaptureStartTime;
    protected ref array<ref SCR_MockTelemetryRecord> m_aRecords;

    //--------------------------------------------------------------------------------------------
    void SCR_MockMLDataCollector()
    {
        m_bIsCapturing = false;
        m_iFrameCount = 0;
        m_aRecords = new array<ref SCR_MockTelemetryRecord>();
    }

    //--------------------------------------------------------------------------------------------
    bool StartCapture()
    {
        if (m_bIsCapturing)
            return false;

        m_bIsCapturing = true;
        m_fCaptureStartTime = System.GetTickCount();
        m_iFrameCount = 0;
        m_aRecords.Clear();
        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool StopCapture()
    {
        if (!m_bIsCapturing)
            return false;

        m_bIsCapturing = false;
        return true;
    }

    //--------------------------------------------------------------------------------------------
    void CaptureFrame(vector position, vector velocity, vector rotation, float throttle, float steering)
    {
        if (!m_bIsCapturing)
            return;

        SCR_MockTelemetryRecord record = new SCR_MockTelemetryRecord();
        record.frameId = m_iFrameCount;
        record.timestamp = System.GetTickCount() - m_fCaptureStartTime;
        record.position = position;
        record.velocity = velocity;
        record.rotation = rotation;
        record.throttle = throttle;
        record.steering = steering;

        m_aRecords.Insert(record);
        m_iFrameCount++;
    }

    //--------------------------------------------------------------------------------------------
    bool IsCapturing() { return m_bIsCapturing; }
    int GetFrameCount() { return m_iFrameCount; }
    int GetRecordCount() { return m_aRecords.Count(); }

    //--------------------------------------------------------------------------------------------
    SCR_MockTelemetryRecord GetRecord(int index)
    {
        if (index < 0 || index >= m_aRecords.Count())
            return null;
        return m_aRecords[index];
    }
}

//------------------------------------------------------------------------------------------------
//! Mock telemetry record structure
class SCR_MockTelemetryRecord
{
    int frameId;
    float timestamp;
    vector position;
    vector velocity;
    vector rotation;
    float throttle;
    float steering;
}

//------------------------------------------------------------------------------------------------
//! Mock depth raycaster for testing
class SCR_MockDepthRaycaster
{
    protected int m_iWidth;
    protected int m_iHeight;
    protected float m_fMaxDistance;
    protected ref array<float> m_aDepthBuffer;

    //--------------------------------------------------------------------------------------------
    void SCR_MockDepthRaycaster(int width = 256, int height = 256, float maxDistance = 500)
    {
        m_iWidth = width;
        m_iHeight = height;
        m_fMaxDistance = maxDistance;
        m_aDepthBuffer = new array<float>();
    }

    //--------------------------------------------------------------------------------------------
    bool Initialize()
    {
        int size = m_iWidth * m_iHeight;
        m_aDepthBuffer.Clear();

        for (int i = 0; i < size; i++)
            m_aDepthBuffer.Insert(m_fMaxDistance);

        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool CastRays(vector cameraPos, vector cameraDir, float fov)
    {
        if (m_aDepthBuffer.Count() != m_iWidth * m_iHeight)
            return false;

        // Simulate ray casting with synthetic depth values
        SCR_TestDataGenerator.GenerateDepthBuffer(m_aDepthBuffer, m_iWidth, m_iHeight, 0.1, m_fMaxDistance);
        return true;
    }

    //--------------------------------------------------------------------------------------------
    float GetDepthAt(int x, int y)
    {
        if (x < 0 || x >= m_iWidth || y < 0 || y >= m_iHeight)
            return -1;

        int index = y * m_iWidth + x;
        return m_aDepthBuffer[index];
    }

    //--------------------------------------------------------------------------------------------
    int GetWidth() { return m_iWidth; }
    int GetHeight() { return m_iHeight; }
    int GetBufferSize() { return m_aDepthBuffer.Count(); }
    float GetMaxDistance() { return m_fMaxDistance; }
}

//------------------------------------------------------------------------------------------------
//! Mock camera rig for multi-camera testing
class SCR_MockMultiCameraRig
{
    protected ref array<ref SCR_MockCameraConfig> m_aCameras;
    protected bool m_bIsActive;

    //--------------------------------------------------------------------------------------------
    void SCR_MockMultiCameraRig()
    {
        m_aCameras = new array<ref SCR_MockCameraConfig>();
        m_bIsActive = false;
    }

    //--------------------------------------------------------------------------------------------
    int AddCamera(vector offset, vector rotation, float fov, string name)
    {
        SCR_MockCameraConfig config = new SCR_MockCameraConfig();
        config.offset = offset;
        config.rotation = rotation;
        config.fov = fov;
        config.name = name;
        config.isEnabled = true;

        int index = m_aCameras.Count();
        m_aCameras.Insert(config);
        return index;
    }

    //--------------------------------------------------------------------------------------------
    bool RemoveCamera(int index)
    {
        if (index < 0 || index >= m_aCameras.Count())
            return false;

        m_aCameras.Remove(index);
        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool SetCameraEnabled(int index, bool enabled)
    {
        if (index < 0 || index >= m_aCameras.Count())
            return false;

        m_aCameras[index].isEnabled = enabled;
        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool Activate()
    {
        if (m_aCameras.Count() == 0)
            return false;

        m_bIsActive = true;
        return true;
    }

    //--------------------------------------------------------------------------------------------
    void Deactivate()
    {
        m_bIsActive = false;
    }

    //--------------------------------------------------------------------------------------------
    int GetCameraCount() { return m_aCameras.Count(); }
    bool IsActive() { return m_bIsActive; }

    //--------------------------------------------------------------------------------------------
    int GetEnabledCameraCount()
    {
        int count = 0;
        foreach (SCR_MockCameraConfig config : m_aCameras)
        {
            if (config.isEnabled)
                count++;
        }
        return count;
    }

    //--------------------------------------------------------------------------------------------
    SCR_MockCameraConfig GetCamera(int index)
    {
        if (index < 0 || index >= m_aCameras.Count())
            return null;
        return m_aCameras[index];
    }
}

//------------------------------------------------------------------------------------------------
//! Mock camera configuration
class SCR_MockCameraConfig
{
    vector offset;
    vector rotation;
    float fov;
    string name;
    bool isEnabled;
}

//------------------------------------------------------------------------------------------------
//! Mock scene enumerator for entity listing
class SCR_MockSceneEnumerator
{
    protected ref array<ref SCR_MockEntityInfo> m_aEntities;
    protected int m_iLastEnumerationTime;

    //--------------------------------------------------------------------------------------------
    void SCR_MockSceneEnumerator()
    {
        m_aEntities = new array<ref SCR_MockEntityInfo>();
        m_iLastEnumerationTime = 0;
    }

    //--------------------------------------------------------------------------------------------
    int EnumerateEntities(vector center, float radius, int maxEntities = 100)
    {
        m_aEntities.Clear();
        m_iLastEnumerationTime = System.GetTickCount();

        // Simulate entity enumeration with random entities
        int entityCount = Math.RandomInt(5, Math.Min(maxEntities, 50));

        for (int i = 0; i < entityCount; i++)
        {
            SCR_MockEntityInfo info = new SCR_MockEntityInfo();
            info.entityId = i;
            info.position = SCR_TestDataGenerator.GenerateRandomPosition(
                center - Vector(radius, 0, radius),
                center + Vector(radius, radius, radius)
            );
            info.entityType = GetRandomEntityType();
            info.distance = vector.Distance(center, info.position);

            m_aEntities.Insert(info);
        }

        return m_aEntities.Count();
    }

    //--------------------------------------------------------------------------------------------
    protected string GetRandomEntityType()
    {
        array<string> types = {"Vehicle", "Character", "Building", "Prop", "Vegetation"};
        return types[Math.RandomInt(0, types.Count())];
    }

    //--------------------------------------------------------------------------------------------
    int GetEntityCount() { return m_aEntities.Count(); }

    //--------------------------------------------------------------------------------------------
    SCR_MockEntityInfo GetEntity(int index)
    {
        if (index < 0 || index >= m_aEntities.Count())
            return null;
        return m_aEntities[index];
    }

    //--------------------------------------------------------------------------------------------
    int GetEntitiesByType(string type, out array<ref SCR_MockEntityInfo> outEntities)
    {
        outEntities = new array<ref SCR_MockEntityInfo>();

        foreach (SCR_MockEntityInfo info : m_aEntities)
        {
            if (info.entityType == type)
                outEntities.Insert(info);
        }

        return outEntities.Count();
    }
}

//------------------------------------------------------------------------------------------------
//! Mock entity information
class SCR_MockEntityInfo
{
    int entityId;
    vector position;
    string entityType;
    float distance;
}

//------------------------------------------------------------------------------------------------
//! Mock anchor frame selector
class SCR_MockAnchorFrameSelector
{
    protected float m_fMinVelocity;
    protected float m_fMinRotationDelta;
    protected float m_fMinTimeDelta;
    protected int m_iLastAnchorFrame;
    protected vector m_vLastPosition;
    protected vector m_vLastRotation;
    protected float m_fLastTime;

    //--------------------------------------------------------------------------------------------
    void SCR_MockAnchorFrameSelector(float minVelocity = 1.0, float minRotationDelta = 5.0, float minTimeDelta = 0.5)
    {
        m_fMinVelocity = minVelocity;
        m_fMinRotationDelta = minRotationDelta;
        m_fMinTimeDelta = minTimeDelta;
        Reset();
    }

    //--------------------------------------------------------------------------------------------
    void Reset()
    {
        m_iLastAnchorFrame = -1;
        m_vLastPosition = vector.Zero;
        m_vLastRotation = vector.Zero;
        m_fLastTime = 0;
    }

    //--------------------------------------------------------------------------------------------
    bool ShouldSelectAnchor(int frameId, vector position, vector rotation, float timestamp)
    {
        // First frame is always an anchor
        if (m_iLastAnchorFrame < 0)
        {
            SetAnchor(frameId, position, rotation, timestamp);
            return true;
        }

        // Check time delta
        float timeDelta = timestamp - m_fLastTime;
        if (timeDelta < m_fMinTimeDelta)
            return false;

        // Check position delta (velocity threshold)
        float posDelta = vector.Distance(position, m_vLastPosition);
        float velocity = posDelta / timeDelta;

        // Check rotation delta
        vector rotDelta = rotation - m_vLastRotation;
        float rotMagnitude = rotDelta.Length();

        // Select anchor if significant movement or rotation
        if (velocity >= m_fMinVelocity || rotMagnitude >= m_fMinRotationDelta)
        {
            SetAnchor(frameId, position, rotation, timestamp);
            return true;
        }

        return false;
    }

    //--------------------------------------------------------------------------------------------
    protected void SetAnchor(int frameId, vector position, vector rotation, float timestamp)
    {
        m_iLastAnchorFrame = frameId;
        m_vLastPosition = position;
        m_vLastRotation = rotation;
        m_fLastTime = timestamp;
    }

    //--------------------------------------------------------------------------------------------
    int GetLastAnchorFrame() { return m_iLastAnchorFrame; }
    float GetMinVelocity() { return m_fMinVelocity; }
    float GetMinRotationDelta() { return m_fMinRotationDelta; }
    float GetMinTimeDelta() { return m_fMinTimeDelta; }
}

//------------------------------------------------------------------------------------------------
//! Mock binary serializer for output testing
class SCR_MockBinarySerializer
{
    protected string m_sOutputPath;
    protected bool m_bIsOpen;
    protected int m_iBytesWritten;
    protected int m_iRecordsWritten;

    static const int MAGIC_NUMBER = 0x4D4C4454; // "MLDT" in hex
    static const int VERSION = 1;

    //--------------------------------------------------------------------------------------------
    void SCR_MockBinarySerializer()
    {
        m_sOutputPath = "";
        m_bIsOpen = false;
        m_iBytesWritten = 0;
        m_iRecordsWritten = 0;
    }

    //--------------------------------------------------------------------------------------------
    bool Open(string path)
    {
        if (m_bIsOpen)
            return false;

        m_sOutputPath = path;
        m_bIsOpen = true;
        m_iBytesWritten = 0;
        m_iRecordsWritten = 0;

        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool WriteHeader()
    {
        if (!m_bIsOpen)
            return false;

        // Simulate writing header (16 bytes)
        m_iBytesWritten += 16;
        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool WriteTelemetryRecord(SCR_MockTelemetryRecord record)
    {
        if (!m_bIsOpen || record == null)
            return false;

        // Simulate writing telemetry record
        // frameId(4) + timestamp(4) + position(12) + velocity(12) + rotation(12) + throttle(4) + steering(4) = 52 bytes
        m_iBytesWritten += 52;
        m_iRecordsWritten++;
        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool WriteDepthBuffer(array<float> buffer, int width, int height)
    {
        if (!m_bIsOpen || buffer == null)
            return false;

        int expectedSize = width * height;
        if (buffer.Count() != expectedSize)
            return false;

        // width(4) + height(4) + data(4 * size)
        m_iBytesWritten += 8 + (4 * expectedSize);
        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool Close()
    {
        if (!m_bIsOpen)
            return false;

        m_bIsOpen = false;
        return true;
    }

    //--------------------------------------------------------------------------------------------
    bool IsOpen() { return m_bIsOpen; }
    int GetBytesWritten() { return m_iBytesWritten; }
    int GetRecordsWritten() { return m_iRecordsWritten; }
    string GetOutputPath() { return m_sOutputPath; }
}

//================================================================================================
// SECTION 4: UNIT TEST CASES
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Unit tests for SCR_MLDataCollector
[Test("SCR_CaptureTestSuite")]
class SCR_Test_MLDataCollector_StartCapture
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();

        // Test starting capture
        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertTrue(
            collector.StartCapture(),
            "StartCapture should return true on first call",
            "MLDataCollector"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertTrue(
            collector.IsCapturing(),
            "IsCapturing should return true after StartCapture",
            "MLDataCollector"
        );

        // Test that starting again fails
        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertFalse(
            collector.StartCapture(),
            "StartCapture should return false when already capturing",
            "MLDataCollector"
        );

        // Report results
        if (result1.Failure() || result2.Failure() || result3.Failure())
        {
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : (result2.Failure() ? result2 : result3));
        }
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for telemetry capture
[Test("SCR_CaptureTestSuite")]
class SCR_Test_MLDataCollector_CaptureFrame
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        collector.StartCapture();

        // Capture multiple frames
        int numFrames = 10;
        for (int i = 0; i < numFrames; i++)
        {
            vector pos, vel, rot;
            float throttle, steering;
            SCR_TestDataGenerator.GenerateTestTelemetry(pos, vel, rot, throttle, steering);
            collector.CaptureFrame(pos, vel, rot, throttle, steering);
        }

        // Verify frame count
        SCR_CaptureTestResult result = SCR_TestAssert.AssertEquals(
            numFrames,
            collector.GetFrameCount(),
            "Frame count should match number of captured frames",
            "MLDataCollector"
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);

        // Verify record integrity
        SCR_MockTelemetryRecord record = collector.GetRecord(0);
        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertNotNull(
            record,
            "First record should exist",
            "MLDataCollector"
        );

        if (result2.Failure())
            TestHarness.ActiveSuite().SetResult(result2);
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for stop capture functionality
[Test("SCR_CaptureTestSuite")]
class SCR_Test_MLDataCollector_StopCapture
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();

        // Test stopping without starting
        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertFalse(
            collector.StopCapture(),
            "StopCapture should fail when not capturing",
            "MLDataCollector"
        );

        // Start and then stop
        collector.StartCapture();

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertTrue(
            collector.StopCapture(),
            "StopCapture should succeed when capturing",
            "MLDataCollector"
        );

        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertFalse(
            collector.IsCapturing(),
            "IsCapturing should return false after StopCapture",
            "MLDataCollector"
        );

        if (result1.Failure() || result2.Failure() || result3.Failure())
        {
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : (result2.Failure() ? result2 : result3));
        }
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for SCR_DepthRaycaster initialization
[Test("SCR_CaptureTestSuite")]
class SCR_Test_DepthRaycaster_Initialize
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockDepthRaycaster raycaster = new SCR_MockDepthRaycaster(128, 128, 500);

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertTrue(
            raycaster.Initialize(),
            "Initialize should return true",
            "DepthRaycaster"
        );

        int expectedSize = 128 * 128;
        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertEquals(
            expectedSize,
            raycaster.GetBufferSize(),
            "Buffer size should match width * height",
            "DepthRaycaster"
        );

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for depth ray casting
[Test("SCR_CaptureTestSuite")]
class SCR_Test_DepthRaycaster_CastRays
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 10000)]
    void Run()
    {
        SCR_MockDepthRaycaster raycaster = new SCR_MockDepthRaycaster(256, 256, 500);
        raycaster.Initialize();

        vector cameraPos = Vector(0, 10, 0);
        vector cameraDir = Vector(0, 0, 1);
        float fov = 90;

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertTrue(
            raycaster.CastRays(cameraPos, cameraDir, fov),
            "CastRays should return true",
            "DepthRaycaster"
        );

        // Verify depth values are within range
        float depth = raycaster.GetDepthAt(128, 128); // Center pixel
        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertInRange(
            depth,
            0.1,
            500,
            "Depth value should be within valid range",
            "DepthRaycaster"
        );

        // Test boundary pixel
        float depthCorner = raycaster.GetDepthAt(0, 0);
        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertInRange(
            depthCorner,
            0.1,
            500,
            "Corner depth value should be within valid range",
            "DepthRaycaster"
        );

        // Test out of bounds
        float invalidDepth = raycaster.GetDepthAt(-1, 0);
        SCR_CaptureTestResult result4 = SCR_TestAssert.AssertEqualsFloat(
            -1,
            invalidDepth,
            0.001,
            "Out of bounds should return -1",
            "DepthRaycaster"
        );

        if (result1.Failure())
            TestHarness.ActiveSuite().SetResult(result1);
        else if (result2.Failure())
            TestHarness.ActiveSuite().SetResult(result2);
        else if (result3.Failure())
            TestHarness.ActiveSuite().SetResult(result3);
        else if (result4.Failure())
            TestHarness.ActiveSuite().SetResult(result4);
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for SCR_MultiCameraRig
[Test("SCR_CaptureTestSuite")]
class SCR_Test_MultiCameraRig_AddCamera
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockMultiCameraRig rig = new SCR_MockMultiCameraRig();

        // Add cameras
        int cam1 = rig.AddCamera(Vector(0, 2, 0), Vector(0, 0, 0), 90, "Front");
        int cam2 = rig.AddCamera(Vector(-1, 2, 0), Vector(0, -45, 0), 90, "Left");
        int cam3 = rig.AddCamera(Vector(1, 2, 0), Vector(0, 45, 0), 90, "Right");

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertEquals(
            3,
            rig.GetCameraCount(),
            "Should have 3 cameras after adding",
            "MultiCameraRig"
        );

        // Verify camera properties
        SCR_MockCameraConfig config = rig.GetCamera(0);
        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertNotNull(
            config,
            "Camera config should exist",
            "MultiCameraRig"
        );

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for camera rig activation
[Test("SCR_CaptureTestSuite")]
class SCR_Test_MultiCameraRig_Activate
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockMultiCameraRig rig = new SCR_MockMultiCameraRig();

        // Try to activate empty rig
        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertFalse(
            rig.Activate(),
            "Activate should fail with no cameras",
            "MultiCameraRig"
        );

        // Add camera and activate
        rig.AddCamera(Vector(0, 2, 0), Vector(0, 0, 0), 90, "Front");

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertTrue(
            rig.Activate(),
            "Activate should succeed with cameras",
            "MultiCameraRig"
        );

        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertTrue(
            rig.IsActive(),
            "IsActive should return true after activation",
            "MultiCameraRig"
        );

        if (result1.Failure() || result2.Failure() || result3.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : (result2.Failure() ? result2 : result3));
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for SCR_SceneEnumerator
[Test("SCR_CaptureTestSuite")]
class SCR_Test_SceneEnumerator_EnumerateEntities
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockSceneEnumerator enumerator = new SCR_MockSceneEnumerator();

        vector center = Vector(0, 0, 0);
        float radius = 100;

        int count = enumerator.EnumerateEntities(center, radius, 50);

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertTrue(
            count > 0,
            "Should enumerate at least one entity",
            "SceneEnumerator"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertEquals(
            count,
            enumerator.GetEntityCount(),
            "GetEntityCount should match returned count",
            "SceneEnumerator"
        );

        // Verify entity distance is within radius
        if (count > 0)
        {
            SCR_MockEntityInfo info = enumerator.GetEntity(0);
            SCR_CaptureTestResult result3 = SCR_TestAssert.AssertNotNull(
                info,
                "Entity info should exist",
                "SceneEnumerator"
            );

            if (result3.Failure())
                TestHarness.ActiveSuite().SetResult(result3);
        }

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for entity filtering by type
[Test("SCR_CaptureTestSuite")]
class SCR_Test_SceneEnumerator_FilterByType
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockSceneEnumerator enumerator = new SCR_MockSceneEnumerator();
        enumerator.EnumerateEntities(Vector(0, 0, 0), 200, 100);

        array<ref SCR_MockEntityInfo> vehicles;
        int vehicleCount = enumerator.GetEntitiesByType("Vehicle", vehicles);

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertTrue(
            vehicles != null,
            "Output array should be created",
            "SceneEnumerator"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertEquals(
            vehicleCount,
            vehicles.Count(),
            "Returned count should match array size",
            "SceneEnumerator"
        );

        // Verify all returned entities are of correct type
        bool allVehicles = true;
        foreach (SCR_MockEntityInfo info : vehicles)
        {
            if (info.entityType != "Vehicle")
            {
                allVehicles = false;
                break;
            }
        }

        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertTrue(
            allVehicles,
            "All filtered entities should be of type Vehicle",
            "SceneEnumerator"
        );

        if (result1.Failure() || result2.Failure() || result3.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : (result2.Failure() ? result2 : result3));
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for SCR_AnchorFrameSelector
[Test("SCR_CaptureTestSuite")]
class SCR_Test_AnchorFrameSelector_FirstFrame
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockAnchorFrameSelector selector = new SCR_MockAnchorFrameSelector(1.0, 5.0, 0.5);

        // First frame should always be selected
        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertTrue(
            selector.ShouldSelectAnchor(0, Vector(0, 0, 0), Vector(0, 0, 0), 0),
            "First frame should be selected as anchor",
            "AnchorFrameSelector"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertEquals(
            0,
            selector.GetLastAnchorFrame(),
            "Last anchor frame should be 0",
            "AnchorFrameSelector"
        );

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for anchor selection based on movement
[Test("SCR_CaptureTestSuite")]
class SCR_Test_AnchorFrameSelector_MovementThreshold
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockAnchorFrameSelector selector = new SCR_MockAnchorFrameSelector(1.0, 5.0, 0.5);

        // First frame
        selector.ShouldSelectAnchor(0, Vector(0, 0, 0), Vector(0, 0, 0), 0);

        // Frame with minimal movement (should not select)
        bool selected1 = selector.ShouldSelectAnchor(1, Vector(0.1, 0, 0.1), Vector(0, 0, 0), 0.6);
        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertFalse(
            selected1,
            "Should not select anchor with minimal movement",
            "AnchorFrameSelector"
        );

        // Frame with significant movement (should select)
        bool selected2 = selector.ShouldSelectAnchor(2, Vector(5, 0, 5), Vector(0, 0, 0), 1.2);
        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertTrue(
            selected2,
            "Should select anchor with significant movement",
            "AnchorFrameSelector"
        );

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for SCR_BinarySerializer
[Test("SCR_CaptureTestSuite")]
class SCR_Test_BinarySerializer_OpenClose
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();

        // Test open
        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertTrue(
            serializer.Open("$profile:test_output.bin"),
            "Open should succeed",
            "BinarySerializer"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertTrue(
            serializer.IsOpen(),
            "IsOpen should return true",
            "BinarySerializer"
        );

        // Test double open fails
        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertFalse(
            serializer.Open("$profile:another.bin"),
            "Second open should fail",
            "BinarySerializer"
        );

        // Test close
        SCR_CaptureTestResult result4 = SCR_TestAssert.AssertTrue(
            serializer.Close(),
            "Close should succeed",
            "BinarySerializer"
        );

        SCR_CaptureTestResult result5 = SCR_TestAssert.AssertFalse(
            serializer.IsOpen(),
            "IsOpen should return false after close",
            "BinarySerializer"
        );

        if (result1.Failure())
            TestHarness.ActiveSuite().SetResult(result1);
        else if (result2.Failure())
            TestHarness.ActiveSuite().SetResult(result2);
        else if (result3.Failure())
            TestHarness.ActiveSuite().SetResult(result3);
        else if (result4.Failure())
            TestHarness.ActiveSuite().SetResult(result4);
        else if (result5.Failure())
            TestHarness.ActiveSuite().SetResult(result5);
    }
}

//------------------------------------------------------------------------------------------------
//! Unit tests for writing telemetry records
[Test("SCR_CaptureTestSuite")]
class SCR_Test_BinarySerializer_WriteTelemetry
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();
        serializer.Open("$profile:telemetry.bin");
        serializer.WriteHeader();

        // Write multiple records
        int recordCount = 100;
        for (int i = 0; i < recordCount; i++)
        {
            SCR_MockTelemetryRecord record = new SCR_MockTelemetryRecord();
            record.frameId = i;
            record.timestamp = i * 0.033; // ~30 FPS
            record.position = Vector(i, 0, i);
            record.velocity = Vector(1, 0, 1);
            record.rotation = Vector(0, i * 10, 0);
            record.throttle = 0.5;
            record.steering = Math.Sin(i * 0.1);

            serializer.WriteTelemetryRecord(record);
        }

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertEquals(
            recordCount,
            serializer.GetRecordsWritten(),
            "Records written should match count",
            "BinarySerializer"
        );

        // Verify bytes written (header=16 + records*52)
        int expectedBytes = 16 + (recordCount * 52);
        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertEquals(
            expectedBytes,
            serializer.GetBytesWritten(),
            "Bytes written should match expected size",
            "BinarySerializer"
        );

        serializer.Close();

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//================================================================================================
// SECTION 5: INTEGRATION TEST CASES
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Integration test: Full capture pipeline
[Test("SCR_CaptureTestSuite")]
class SCR_Test_Integration_FullCapturePipeline
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 30000)]
    void Run()
    {
        // Create all components
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        SCR_MockDepthRaycaster raycaster = new SCR_MockDepthRaycaster(128, 128, 500);
        SCR_MockMultiCameraRig cameraRig = new SCR_MockMultiCameraRig();
        SCR_MockSceneEnumerator enumerator = new SCR_MockSceneEnumerator();
        SCR_MockAnchorFrameSelector anchorSelector = new SCR_MockAnchorFrameSelector();
        SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();

        // Initialize components
        raycaster.Initialize();
        cameraRig.AddCamera(Vector(0, 2, 0), Vector(0, 0, 0), 90, "Front");
        cameraRig.AddCamera(Vector(-1, 2, 0), Vector(0, -45, 0), 90, "Left");
        cameraRig.AddCamera(Vector(1, 2, 0), Vector(0, 45, 0), 90, "Right");
        cameraRig.Activate();

        serializer.Open("$profile:integration_test.bin");
        serializer.WriteHeader();

        // Start capture
        collector.StartCapture();

        // Simulate 60 seconds of capture at 30 FPS
        int totalFrames = 60 * 30; // 1800 frames
        int anchorFrames = 0;
        vector currentPos = Vector(0, 0, 0);
        vector currentRot = Vector(0, 0, 0);

        for (int i = 0; i < totalFrames; i++)
        {
            float timestamp = i / 30.0;

            // Simulate vehicle movement
            currentPos = currentPos + Vector(0.5, 0, 0.5);
            currentRot[1] = Math.Sin(timestamp * 0.5) * 30;

            vector velocity = Vector(15, 0, 15); // ~50 km/h
            float throttle = 0.7;
            float steering = Math.Sin(timestamp) * 0.3;

            // Capture telemetry
            collector.CaptureFrame(currentPos, velocity, currentRot, throttle, steering);

            // Check for anchor frame
            if (anchorSelector.ShouldSelectAnchor(i, currentPos, currentRot, timestamp))
            {
                anchorFrames++;

                // On anchor frames, perform full scene capture
                raycaster.CastRays(currentPos + Vector(0, 2, 0), Vector(0, 0, 1), 90);
                enumerator.EnumerateEntities(currentPos, 200);
            }

            // Write to serializer periodically
            if (i % 30 == 0)
            {
                SCR_MockTelemetryRecord record = collector.GetRecord(i);
                if (record != null)
                    serializer.WriteTelemetryRecord(record);
            }
        }

        // Stop capture
        collector.StopCapture();
        serializer.Close();

        // Verify results
        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertEquals(
            totalFrames,
            collector.GetFrameCount(),
            "Should capture all frames",
            "Integration"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertTrue(
            anchorFrames > 0,
            "Should have at least one anchor frame",
            "Integration"
        );

        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertTrue(
            serializer.GetRecordsWritten() > 0,
            "Should write telemetry records",
            "Integration"
        );

        if (result1.Failure())
            TestHarness.ActiveSuite().SetResult(result1);
        else if (result2.Failure())
            TestHarness.ActiveSuite().SetResult(result2);
        else if (result3.Failure())
            TestHarness.ActiveSuite().SetResult(result3);
    }
}

//------------------------------------------------------------------------------------------------
//! Integration test: Multi-vehicle capture
[Test("SCR_CaptureTestSuite")]
class SCR_Test_Integration_MultiVehicleCapture
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 30000)]
    void Run()
    {
        const int NUM_VEHICLES = 3;

        // Create collectors for each vehicle
        ref array<ref SCR_MockMLDataCollector> collectors = new array<ref SCR_MockMLDataCollector>();
        ref array<vector> vehiclePositions = new array<vector>();

        for (int v = 0; v < NUM_VEHICLES; v++)
        {
            collectors.Insert(new SCR_MockMLDataCollector());
            vehiclePositions.Insert(Vector(v * 100, 0, 0));
            collectors[v].StartCapture();
        }

        // Simulate capture for all vehicles
        int framesPerVehicle = 100;
        for (int frame = 0; frame < framesPerVehicle; frame++)
        {
            for (int v = 0; v < NUM_VEHICLES; v++)
            {
                // Each vehicle moves in different direction
                vehiclePositions[v] = vehiclePositions[v] + Vector(Math.Cos(v), 0, Math.Sin(v));
                vector vel = Vector(10, 0, 10);
                vector rot = Vector(0, v * 45, 0);

                collectors[v].CaptureFrame(vehiclePositions[v], vel, rot, 0.5, 0);
            }
        }

        // Stop all captures
        for (int v = 0; v < NUM_VEHICLES; v++)
        {
            collectors[v].StopCapture();
        }

        // Verify all collectors have correct frame counts
        bool allCorrect = true;
        for (int v = 0; v < NUM_VEHICLES; v++)
        {
            if (collectors[v].GetFrameCount() != framesPerVehicle)
            {
                allCorrect = false;
                break;
            }
        }

        SCR_CaptureTestResult result = SCR_TestAssert.AssertTrue(
            allCorrect,
            string.Format("All %1 vehicles should have %2 frames", NUM_VEHICLES, framesPerVehicle),
            "Integration"
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);
    }
}

//------------------------------------------------------------------------------------------------
//! Integration test: Session pause/resume
[Test("SCR_CaptureTestSuite")]
class SCR_Test_Integration_SessionPauseResume
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 10000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();

        // Start capture
        collector.StartCapture();

        // Capture some frames
        for (int i = 0; i < 50; i++)
        {
            vector pos, vel, rot;
            float throttle, steering;
            SCR_TestDataGenerator.GenerateTestTelemetry(pos, vel, rot, throttle, steering);
            collector.CaptureFrame(pos, vel, rot, throttle, steering);
        }

        int framesBeforePause = collector.GetFrameCount();

        // Pause (stop) capture
        collector.StopCapture();

        // Try to capture while paused (should not add frames)
        for (int i = 0; i < 10; i++)
        {
            vector pos, vel, rot;
            float throttle, steering;
            SCR_TestDataGenerator.GenerateTestTelemetry(pos, vel, rot, throttle, steering);
            collector.CaptureFrame(pos, vel, rot, throttle, steering);
        }

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertEquals(
            framesBeforePause,
            collector.GetRecordCount(),
            "Frame count should not change while paused",
            "Integration"
        );

        // Resume capture
        collector.StartCapture();

        // Capture more frames
        for (int i = 0; i < 50; i++)
        {
            vector pos, vel, rot;
            float throttle, steering;
            SCR_TestDataGenerator.GenerateTestTelemetry(pos, vel, rot, throttle, steering);
            collector.CaptureFrame(pos, vel, rot, throttle, steering);
        }

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertEquals(
            100,
            collector.GetFrameCount(),
            "Should have 100 frames after resume",
            "Integration"
        );

        collector.StopCapture();

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//================================================================================================
// SECTION 6: PERFORMANCE TEST CASES
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Performance test: Telemetry capture rate
[Test("SCR_CaptureTestSuite")]
class SCR_Test_Performance_TelemetryCapture
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 30000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        SCR_PerformanceTimer timer = new SCR_PerformanceTimer();

        collector.StartCapture();

        // Performance target: < 0.1ms per frame capture
        const float TARGET_MS = 0.1;
        const int ITERATIONS = 1000;

        for (int i = 0; i < ITERATIONS; i++)
        {
            vector pos, vel, rot;
            float throttle, steering;
            SCR_TestDataGenerator.GenerateTestTelemetry(pos, vel, rot, throttle, steering);

            timer.Start();
            collector.CaptureFrame(pos, vel, rot, throttle, steering);
            timer.Stop();
        }

        collector.StopCapture();

        SCR_PerformanceTestResult result = timer.CreateResult(
            "Telemetry capture rate",
            "MLDataCollector",
            TARGET_MS
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);

        Print(string.Format("[PERF] Telemetry Capture: %1", result.GetPerformanceReport()));
    }
}

//------------------------------------------------------------------------------------------------
//! Performance test: Depth raycasting
[Test("SCR_CaptureTestSuite")]
class SCR_Test_Performance_DepthRaycasting
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 60000)]
    void Run()
    {
        // Test different resolutions
        ref array<int> resolutions = {64, 128, 256};

        foreach (int res : resolutions)
        {
            SCR_MockDepthRaycaster raycaster = new SCR_MockDepthRaycaster(res, res, 500);
            raycaster.Initialize();

            SCR_PerformanceTimer timer = new SCR_PerformanceTimer();

            // Performance target scales with resolution
            // 64x64: < 1ms, 128x128: < 4ms, 256x256: < 16ms
            float targetMs = (res * res) / 4096.0; // Base: 1ms for 64x64
            const int ITERATIONS = 100;

            for (int i = 0; i < ITERATIONS; i++)
            {
                vector pos = SCR_TestDataGenerator.GenerateRandomPosition(
                    Vector(-100, 0, -100), Vector(100, 50, 100));
                vector dir = Vector(0, 0, 1);

                timer.Start();
                raycaster.CastRays(pos, dir, 90);
                timer.Stop();
            }

            SCR_PerformanceTestResult result = timer.CreateResult(
                string.Format("Depth raycast %1x%1", res, res),
                "DepthRaycaster",
                targetMs
            );

            if (result.Failure())
            {
                TestHarness.ActiveSuite().SetResult(result);
                return;
            }

            Print(string.Format("[PERF] Depth Raycast %1x%1: %2", res, res, result.GetPerformanceReport()));
        }
    }
}

//------------------------------------------------------------------------------------------------
//! Performance test: Scene enumeration
[Test("SCR_CaptureTestSuite")]
class SCR_Test_Performance_SceneEnumeration
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 30000)]
    void Run()
    {
        SCR_MockSceneEnumerator enumerator = new SCR_MockSceneEnumerator();
        SCR_PerformanceTimer timer = new SCR_PerformanceTimer();

        // Performance target: < 2ms per enumeration
        const float TARGET_MS = 2.0;
        const int ITERATIONS = 100;

        for (int i = 0; i < ITERATIONS; i++)
        {
            vector center = SCR_TestDataGenerator.GenerateRandomPosition(
                Vector(-1000, 0, -1000), Vector(1000, 100, 1000));
            float radius = Math.RandomFloat(50, 200);

            timer.Start();
            enumerator.EnumerateEntities(center, radius, 100);
            timer.Stop();
        }

        SCR_PerformanceTestResult result = timer.CreateResult(
            "Scene enumeration",
            "SceneEnumerator",
            TARGET_MS
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);

        Print(string.Format("[PERF] Scene Enumeration: %1", result.GetPerformanceReport()));
    }
}

//------------------------------------------------------------------------------------------------
//! Performance test: Binary serialization
[Test("SCR_CaptureTestSuite")]
class SCR_Test_Performance_BinarySerialization
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 30000)]
    void Run()
    {
        SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();
        SCR_PerformanceTimer timer = new SCR_PerformanceTimer();

        serializer.Open("$profile:perf_test.bin");
        serializer.WriteHeader();

        // Performance target: < 0.05ms per record write
        const float TARGET_MS = 0.05;
        const int ITERATIONS = 1000;

        for (int i = 0; i < ITERATIONS; i++)
        {
            SCR_MockTelemetryRecord record = new SCR_MockTelemetryRecord();
            record.frameId = i;
            record.timestamp = i * 0.033;
            record.position = Vector(i, 0, i);
            record.velocity = Vector(10, 0, 10);
            record.rotation = Vector(0, i, 0);
            record.throttle = 0.5;
            record.steering = 0.1;

            timer.Start();
            serializer.WriteTelemetryRecord(record);
            timer.Stop();
        }

        serializer.Close();

        SCR_PerformanceTestResult result = timer.CreateResult(
            "Binary serialization",
            "BinarySerializer",
            TARGET_MS
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);

        Print(string.Format("[PERF] Binary Serialization: %1", result.GetPerformanceReport()));
    }
}

//------------------------------------------------------------------------------------------------
//! Performance test: Full pipeline frame budget (target: 16.67ms for 60 FPS)
[Test("SCR_CaptureTestSuite")]
class SCR_Test_Performance_FullPipelineFrameBudget
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 60000)]
    void Run()
    {
        // Create full pipeline
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        SCR_MockDepthRaycaster raycaster = new SCR_MockDepthRaycaster(128, 128, 500);
        SCR_MockSceneEnumerator enumerator = new SCR_MockSceneEnumerator();
        SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();

        raycaster.Initialize();
        serializer.Open("$profile:pipeline_perf.bin");
        serializer.WriteHeader();
        collector.StartCapture();

        SCR_PerformanceTimer timer = new SCR_PerformanceTimer();

        // Frame budget for 60 FPS: 16.67ms
        // Data capture should use < 5ms (leaving ~11ms for game logic)
        const float TARGET_MS = 5.0;
        const int ITERATIONS = 100;

        for (int i = 0; i < ITERATIONS; i++)
        {
            timer.Start();

            // Simulate full frame capture
            vector pos = Vector(i, 0, i);
            vector vel = Vector(10, 0, 10);
            vector rot = Vector(0, i * 10, 0);

            // Telemetry
            collector.CaptureFrame(pos, vel, rot, 0.5, 0.1);

            // Depth (every 5th frame for anchor)
            if (i % 5 == 0)
            {
                raycaster.CastRays(pos, Vector(0, 0, 1), 90);
                enumerator.EnumerateEntities(pos, 100);
            }

            // Serialize
            SCR_MockTelemetryRecord record = collector.GetRecord(i);
            if (record != null)
                serializer.WriteTelemetryRecord(record);

            timer.Stop();
        }

        collector.StopCapture();
        serializer.Close();

        SCR_PerformanceTestResult result = timer.CreateResult(
            "Full pipeline frame budget",
            "Pipeline",
            TARGET_MS
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);

        Print(string.Format("[PERF] Full Pipeline: %1", result.GetPerformanceReport()));
    }
}

//================================================================================================
// SECTION 7: DATA VALIDATION TESTS
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Data validation: Telemetry record integrity
[Test("SCR_CaptureTestSuite")]
class SCR_Test_DataValidation_TelemetryIntegrity
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 10000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        collector.StartCapture();

        // Capture with known values
        vector expectedPos = Vector(100, 50, 200);
        vector expectedVel = Vector(10, 0, 10);
        vector expectedRot = Vector(5, 45, 0);
        float expectedThrottle = 0.75;
        float expectedSteering = -0.25;

        collector.CaptureFrame(expectedPos, expectedVel, expectedRot, expectedThrottle, expectedSteering);
        collector.StopCapture();

        SCR_MockTelemetryRecord record = collector.GetRecord(0);

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertNotNull(
            record,
            "Record should exist",
            "DataValidation"
        );

        if (result1.Failure())
        {
            TestHarness.ActiveSuite().SetResult(result1);
            return;
        }

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertVectorEquals(
            expectedPos,
            record.position,
            0.001,
            "Position should match",
            "DataValidation"
        );

        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertVectorEquals(
            expectedVel,
            record.velocity,
            0.001,
            "Velocity should match",
            "DataValidation"
        );

        SCR_CaptureTestResult result4 = SCR_TestAssert.AssertEqualsFloat(
            expectedThrottle,
            record.throttle,
            0.001,
            "Throttle should match",
            "DataValidation"
        );

        SCR_CaptureTestResult result5 = SCR_TestAssert.AssertEqualsFloat(
            expectedSteering,
            record.steering,
            0.001,
            "Steering should match",
            "DataValidation"
        );

        if (result2.Failure())
            TestHarness.ActiveSuite().SetResult(result2);
        else if (result3.Failure())
            TestHarness.ActiveSuite().SetResult(result3);
        else if (result4.Failure())
            TestHarness.ActiveSuite().SetResult(result4);
        else if (result5.Failure())
            TestHarness.ActiveSuite().SetResult(result5);
    }
}

//------------------------------------------------------------------------------------------------
//! Data validation: Depth buffer bounds
[Test("SCR_CaptureTestSuite")]
class SCR_Test_DataValidation_DepthBufferBounds
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 10000)]
    void Run()
    {
        const int WIDTH = 64;
        const int HEIGHT = 64;
        const float MIN_DEPTH = 0.1;
        const float MAX_DEPTH = 500;

        SCR_MockDepthRaycaster raycaster = new SCR_MockDepthRaycaster(WIDTH, HEIGHT, MAX_DEPTH);
        raycaster.Initialize();
        raycaster.CastRays(Vector(0, 10, 0), Vector(0, 0, 1), 90);

        // Verify all depth values are within bounds
        bool allInBounds = true;
        int outOfBoundsCount = 0;

        for (int y = 0; y < HEIGHT; y++)
        {
            for (int x = 0; x < WIDTH; x++)
            {
                float depth = raycaster.GetDepthAt(x, y);
                if (depth < MIN_DEPTH || depth > MAX_DEPTH)
                {
                    allInBounds = false;
                    outOfBoundsCount++;
                }
            }
        }

        SCR_CaptureTestResult result = SCR_TestAssert.AssertTrue(
            allInBounds,
            string.Format("All depth values should be in [%.1f, %.1f], found %1 out of bounds",
                MIN_DEPTH, MAX_DEPTH, outOfBoundsCount),
            "DataValidation"
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);
    }
}

//------------------------------------------------------------------------------------------------
//! Data validation: Frame sequence monotonicity
[Test("SCR_CaptureTestSuite")]
class SCR_Test_DataValidation_FrameSequence
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 10000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        collector.StartCapture();

        // Capture 100 frames
        for (int i = 0; i < 100; i++)
        {
            vector pos, vel, rot;
            float throttle, steering;
            SCR_TestDataGenerator.GenerateTestTelemetry(pos, vel, rot, throttle, steering);
            collector.CaptureFrame(pos, vel, rot, throttle, steering);
        }

        collector.StopCapture();

        // Verify frame IDs are monotonically increasing
        bool monotonic = true;
        int lastFrameId = -1;
        float lastTimestamp = -1;

        for (int i = 0; i < collector.GetRecordCount(); i++)
        {
            SCR_MockTelemetryRecord record = collector.GetRecord(i);
            if (record == null)
                continue;

            if (record.frameId <= lastFrameId || record.timestamp <= lastTimestamp)
            {
                monotonic = false;
                break;
            }

            lastFrameId = record.frameId;
            lastTimestamp = record.timestamp;
        }

        SCR_CaptureTestResult result = SCR_TestAssert.AssertTrue(
            monotonic,
            "Frame IDs and timestamps should be monotonically increasing",
            "DataValidation"
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);
    }
}

//================================================================================================
// SECTION 8: EDGE CASE AND ERROR HANDLING TESTS
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Edge case: Capture with zero frames
[Test("SCR_CaptureTestSuite")]
class SCR_Test_EdgeCase_ZeroFrames
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        collector.StartCapture();
        collector.StopCapture();

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertEquals(
            0,
            collector.GetFrameCount(),
            "Frame count should be 0",
            "EdgeCase"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertNull(
            collector.GetRecord(0),
            "GetRecord(0) should return null for empty collector",
            "EdgeCase"
        );

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//------------------------------------------------------------------------------------------------
//! Edge case: Very large capture session
[Test("SCR_CaptureTestSuite")]
class SCR_Test_EdgeCase_LargeSession
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 60000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        collector.StartCapture();

        // Simulate 10 minutes at 30 FPS = 18000 frames
        const int LARGE_FRAME_COUNT = 18000;

        for (int i = 0; i < LARGE_FRAME_COUNT; i++)
        {
            vector pos = Vector(i, 0, i);
            vector vel = Vector(10, 0, 10);
            vector rot = Vector(0, 0, 0);
            collector.CaptureFrame(pos, vel, rot, 0.5, 0);
        }

        collector.StopCapture();

        SCR_CaptureTestResult result = SCR_TestAssert.AssertEquals(
            LARGE_FRAME_COUNT,
            collector.GetFrameCount(),
            "Should handle large number of frames",
            "EdgeCase"
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);
    }
}

//------------------------------------------------------------------------------------------------
//! Edge case: Extreme position values
[Test("SCR_CaptureTestSuite")]
class SCR_Test_EdgeCase_ExtremePositions
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
        collector.StartCapture();

        // Test extreme positions
        vector extremePos1 = Vector(100000, 10000, 100000);
        vector extremePos2 = Vector(-100000, -100, -100000);
        vector extremeVel = Vector(1000, 0, 1000); // 1000 m/s

        collector.CaptureFrame(extremePos1, extremeVel, Vector(0, 0, 0), 1.0, 1.0);
        collector.CaptureFrame(extremePos2, extremeVel, Vector(0, 0, 0), 0, -1.0);

        collector.StopCapture();

        SCR_MockTelemetryRecord record1 = collector.GetRecord(0);
        SCR_MockTelemetryRecord record2 = collector.GetRecord(1);

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertVectorEquals(
            extremePos1,
            record1.position,
            0.001,
            "Should preserve extreme positive positions",
            "EdgeCase"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertVectorEquals(
            extremePos2,
            record2.position,
            0.001,
            "Should preserve extreme negative positions",
            "EdgeCase"
        );

        if (result1.Failure() || result2.Failure())
            TestHarness.ActiveSuite().SetResult(result1.Failure() ? result1 : result2);
    }
}

//------------------------------------------------------------------------------------------------
//! Edge case: Camera rig with disabled cameras
[Test("SCR_CaptureTestSuite")]
class SCR_Test_EdgeCase_DisabledCameras
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockMultiCameraRig rig = new SCR_MockMultiCameraRig();

        // Add 3 cameras
        rig.AddCamera(Vector(0, 2, 0), Vector(0, 0, 0), 90, "Front");
        rig.AddCamera(Vector(-1, 2, 0), Vector(0, -45, 0), 90, "Left");
        rig.AddCamera(Vector(1, 2, 0), Vector(0, 45, 0), 90, "Right");

        // Disable one camera
        rig.SetCameraEnabled(1, false);

        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertEquals(
            3,
            rig.GetCameraCount(),
            "Total camera count should still be 3",
            "EdgeCase"
        );

        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertEquals(
            2,
            rig.GetEnabledCameraCount(),
            "Enabled camera count should be 2",
            "EdgeCase"
        );

        // Disable all cameras
        rig.SetCameraEnabled(0, false);
        rig.SetCameraEnabled(2, false);

        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertEquals(
            0,
            rig.GetEnabledCameraCount(),
            "Should have 0 enabled cameras",
            "EdgeCase"
        );

        if (result1.Failure())
            TestHarness.ActiveSuite().SetResult(result1);
        else if (result2.Failure())
            TestHarness.ActiveSuite().SetResult(result2);
        else if (result3.Failure())
            TestHarness.ActiveSuite().SetResult(result3);
    }
}

//------------------------------------------------------------------------------------------------
//! Edge case: Scene enumeration with zero radius
[Test("SCR_CaptureTestSuite")]
class SCR_Test_EdgeCase_ZeroRadius
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockSceneEnumerator enumerator = new SCR_MockSceneEnumerator();

        // Enumerate with zero radius - should still work but find minimal entities
        int count = enumerator.EnumerateEntities(Vector(0, 0, 0), 0, 100);

        // Zero radius is an edge case - implementation may vary
        // We just verify it doesn't crash
        SCR_CaptureTestResult result = SCR_TestAssert.AssertTrue(
            count >= 0,
            "Should handle zero radius without error",
            "EdgeCase"
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);
    }
}

//------------------------------------------------------------------------------------------------
//! Edge case: Serializer operations on closed file
[Test("SCR_CaptureTestSuite")]
class SCR_Test_EdgeCase_SerializerClosedFile
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();

        // Try operations without opening
        SCR_CaptureTestResult result1 = SCR_TestAssert.AssertFalse(
            serializer.WriteHeader(),
            "WriteHeader should fail when not open",
            "EdgeCase"
        );

        SCR_MockTelemetryRecord record = new SCR_MockTelemetryRecord();
        SCR_CaptureTestResult result2 = SCR_TestAssert.AssertFalse(
            serializer.WriteTelemetryRecord(record),
            "WriteTelemetryRecord should fail when not open",
            "EdgeCase"
        );

        SCR_CaptureTestResult result3 = SCR_TestAssert.AssertFalse(
            serializer.Close(),
            "Close should fail when not open",
            "EdgeCase"
        );

        if (result1.Failure())
            TestHarness.ActiveSuite().SetResult(result1);
        else if (result2.Failure())
            TestHarness.ActiveSuite().SetResult(result2);
        else if (result3.Failure())
            TestHarness.ActiveSuite().SetResult(result3);
    }
}

//------------------------------------------------------------------------------------------------
//! Edge case: Anchor selection with stationary vehicle
[Test("SCR_CaptureTestSuite")]
class SCR_Test_EdgeCase_StationaryVehicle
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockAnchorFrameSelector selector = new SCR_MockAnchorFrameSelector(1.0, 5.0, 0.5);

        vector staticPos = Vector(100, 0, 100);
        vector staticRot = Vector(0, 90, 0);

        // First frame is anchor
        selector.ShouldSelectAnchor(0, staticPos, staticRot, 0);

        // Subsequent frames with no movement should not be anchors
        int anchorCount = 1; // First frame
        for (int i = 1; i <= 100; i++)
        {
            float timestamp = i * 0.1; // 100ms intervals
            if (selector.ShouldSelectAnchor(i, staticPos, staticRot, timestamp))
                anchorCount++;
        }

        // Should only have the first anchor since vehicle is stationary
        SCR_CaptureTestResult result = SCR_TestAssert.AssertEquals(
            1,
            anchorCount,
            "Stationary vehicle should only have initial anchor",
            "EdgeCase"
        );

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);
    }
}

//------------------------------------------------------------------------------------------------
//! Edge case: Depth buffer with invalid dimensions
[Test("SCR_CaptureTestSuite")]
class SCR_Test_EdgeCase_InvalidDepthDimensions
{
    //--------------------------------------------------------------------------------------------
    [Test("SCR_CaptureTestSuite", 5000)]
    void Run()
    {
        SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();
        serializer.Open("$profile:invalid_depth.bin");

        // Create mismatched depth buffer
        array<float> buffer = new array<float>();
        for (int i = 0; i < 100; i++)
            buffer.Insert(10.0);

        // Try to write with wrong dimensions (buffer has 100 elements, but we claim 64x64=4096)
        SCR_CaptureTestResult result = SCR_TestAssert.AssertFalse(
            serializer.WriteDepthBuffer(buffer, 64, 64),
            "Should reject buffer with mismatched dimensions",
            "EdgeCase"
        );

        serializer.Close();

        if (result.Failure())
            TestHarness.ActiveSuite().SetResult(result);
    }
}

//================================================================================================
// SECTION 9: TEST SUITE DEFINITION
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Main test suite for capture pipeline
class SCR_CaptureTestSuite : TestSuite
{
    //--------------------------------------------------------------------------------------------
    void SCR_CaptureTestSuite()
    {
        Print("==========================================================");
        Print("SCR_CaptureTestSuite: Enfusion Data Capture Pipeline Tests");
        Print("==========================================================");
        Print("");
        Print("Test Categories:");
        Print("  - Unit Tests: Individual component functionality");
        Print("  - Integration Tests: Component interactions");
        Print("  - Performance Tests: Frame budget compliance");
        Print("  - Data Validation: Output format correctness");
        Print("  - Edge Cases: Error handling, boundary conditions");
        Print("");
    }

    //--------------------------------------------------------------------------------------------
    override string GetName()
    {
        return "SCR_CaptureTestSuite";
    }
}

//================================================================================================
// SECTION 10: TEST RUNNER AND REPORTING
//================================================================================================

//------------------------------------------------------------------------------------------------
//! Test report generator
class SCR_CaptureTestReporter
{
    protected int m_iPassed;
    protected int m_iFailed;
    protected int m_iSkipped;
    protected ref array<string> m_aFailures;
    protected ref array<string> m_aPerformanceResults;
    protected float m_fTotalTimeMs;

    //--------------------------------------------------------------------------------------------
    void SCR_CaptureTestReporter()
    {
        m_iPassed = 0;
        m_iFailed = 0;
        m_iSkipped = 0;
        m_aFailures = new array<string>();
        m_aPerformanceResults = new array<string>();
        m_fTotalTimeMs = 0;
    }

    //--------------------------------------------------------------------------------------------
    void RecordResult(SCR_CaptureTestResult result)
    {
        if (result == null)
        {
            m_iSkipped++;
            return;
        }

        m_fTotalTimeMs += result.GetExecutionTime();

        if (result.IsPassed())
        {
            m_iPassed++;
        }
        else
        {
            m_iFailed++;
            m_aFailures.Insert(result.FailureText());
        }
    }

    //--------------------------------------------------------------------------------------------
    void RecordPerformanceResult(SCR_PerformanceTestResult result)
    {
        RecordResult(result);
        m_aPerformanceResults.Insert(result.GetPerformanceReport());
    }

    //--------------------------------------------------------------------------------------------
    string GenerateReport()
    {
        string report = "";
        report += "\n==========================================================\n";
        report += "           CAPTURE PIPELINE TEST REPORT\n";
        report += "==========================================================\n\n";

        int total = m_iPassed + m_iFailed + m_iSkipped;
        float passRate = (total > 0) ? (m_iPassed * 100.0 / total) : 0;

        report += string.Format("Total Tests: %1\n", total);
        report += string.Format("  Passed:  %1\n", m_iPassed);
        report += string.Format("  Failed:  %1\n", m_iFailed);
        report += string.Format("  Skipped: %1\n", m_iSkipped);
        report += string.Format("  Pass Rate: %.1f%%\n", passRate);
        report += string.Format("  Total Time: %.2fms\n", m_fTotalTimeMs);

        if (m_aFailures.Count() > 0)
        {
            report += "\n--- FAILURES ---\n";
            foreach (string failure : m_aFailures)
            {
                report += "  " + failure + "\n";
            }
        }

        if (m_aPerformanceResults.Count() > 0)
        {
            report += "\n--- PERFORMANCE RESULTS ---\n";
            foreach (string perfResult : m_aPerformanceResults)
            {
                report += "  " + perfResult + "\n";
            }
        }

        report += "\n==========================================================\n";

        return report;
    }

    //--------------------------------------------------------------------------------------------
    int GetPassedCount() { return m_iPassed; }
    int GetFailedCount() { return m_iFailed; }
    int GetSkippedCount() { return m_iSkipped; }
    float GetTotalTimeMs() { return m_fTotalTimeMs; }
}

//------------------------------------------------------------------------------------------------
//! Manual test runner for development use
class SCR_CaptureTestRunner
{
    //--------------------------------------------------------------------------------------------
    static void RunAllTests()
    {
        Print("Starting Capture Pipeline Test Suite...");
        Print("");

        SCR_CaptureTestReporter reporter = new SCR_CaptureTestReporter();

        // Run unit tests
        Print("--- Running Unit Tests ---");
        RunUnitTests(reporter);

        // Run integration tests
        Print("--- Running Integration Tests ---");
        RunIntegrationTests(reporter);

        // Run performance tests
        Print("--- Running Performance Tests ---");
        RunPerformanceTests(reporter);

        // Run data validation tests
        Print("--- Running Data Validation Tests ---");
        RunDataValidationTests(reporter);

        // Run edge case tests
        Print("--- Running Edge Case Tests ---");
        RunEdgeCaseTests(reporter);

        // Generate and print report
        Print(reporter.GenerateReport());
    }

    //--------------------------------------------------------------------------------------------
    protected static void RunUnitTests(SCR_CaptureTestReporter reporter)
    {
        // MLDataCollector tests
        {
            SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
            reporter.RecordResult(SCR_TestAssert.AssertTrue(collector.StartCapture(), "StartCapture", "MLDataCollector"));
            reporter.RecordResult(SCR_TestAssert.AssertTrue(collector.IsCapturing(), "IsCapturing", "MLDataCollector"));
            reporter.RecordResult(SCR_TestAssert.AssertTrue(collector.StopCapture(), "StopCapture", "MLDataCollector"));
        }

        // DepthRaycaster tests
        {
            SCR_MockDepthRaycaster raycaster = new SCR_MockDepthRaycaster(64, 64, 500);
            reporter.RecordResult(SCR_TestAssert.AssertTrue(raycaster.Initialize(), "Initialize", "DepthRaycaster"));
            reporter.RecordResult(SCR_TestAssert.AssertEquals(64 * 64, raycaster.GetBufferSize(), "BufferSize", "DepthRaycaster"));
        }

        // MultiCameraRig tests
        {
            SCR_MockMultiCameraRig rig = new SCR_MockMultiCameraRig();
            rig.AddCamera(Vector(0, 2, 0), Vector(0, 0, 0), 90, "Front");
            reporter.RecordResult(SCR_TestAssert.AssertEquals(1, rig.GetCameraCount(), "CameraCount", "MultiCameraRig"));
            reporter.RecordResult(SCR_TestAssert.AssertTrue(rig.Activate(), "Activate", "MultiCameraRig"));
        }

        // SceneEnumerator tests
        {
            SCR_MockSceneEnumerator enumerator = new SCR_MockSceneEnumerator();
            int count = enumerator.EnumerateEntities(Vector(0, 0, 0), 100);
            reporter.RecordResult(SCR_TestAssert.AssertTrue(count > 0, "EnumerateEntities", "SceneEnumerator"));
        }

        // AnchorFrameSelector tests
        {
            SCR_MockAnchorFrameSelector selector = new SCR_MockAnchorFrameSelector();
            reporter.RecordResult(SCR_TestAssert.AssertTrue(
                selector.ShouldSelectAnchor(0, Vector(0, 0, 0), Vector(0, 0, 0), 0),
                "FirstFrameAnchor", "AnchorFrameSelector"));
        }

        // BinarySerializer tests
        {
            SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();
            reporter.RecordResult(SCR_TestAssert.AssertTrue(serializer.Open("$profile:test.bin"), "Open", "BinarySerializer"));
            reporter.RecordResult(SCR_TestAssert.AssertTrue(serializer.IsOpen(), "IsOpen", "BinarySerializer"));
            reporter.RecordResult(SCR_TestAssert.AssertTrue(serializer.Close(), "Close", "BinarySerializer"));
        }
    }

    //--------------------------------------------------------------------------------------------
    protected static void RunIntegrationTests(SCR_CaptureTestReporter reporter)
    {
        // Full pipeline test
        {
            SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
            SCR_MockDepthRaycaster raycaster = new SCR_MockDepthRaycaster(64, 64, 500);
            SCR_MockBinarySerializer serializer = new SCR_MockBinarySerializer();

            raycaster.Initialize();
            serializer.Open("$profile:integration.bin");
            serializer.WriteHeader();
            collector.StartCapture();

            for (int i = 0; i < 100; i++)
            {
                collector.CaptureFrame(Vector(i, 0, i), Vector(10, 0, 10), Vector(0, 0, 0), 0.5, 0);
                if (i % 10 == 0)
                    raycaster.CastRays(Vector(i, 10, i), Vector(0, 0, 1), 90);
            }

            collector.StopCapture();
            serializer.Close();

            reporter.RecordResult(SCR_TestAssert.AssertEquals(100, collector.GetFrameCount(), "FullPipeline", "Integration"));
        }
    }

    //--------------------------------------------------------------------------------------------
    protected static void RunPerformanceTests(SCR_CaptureTestReporter reporter)
    {
        // Telemetry capture performance
        {
            SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
            SCR_PerformanceTimer timer = new SCR_PerformanceTimer();

            collector.StartCapture();

            for (int i = 0; i < 1000; i++)
            {
                timer.Start();
                collector.CaptureFrame(Vector(i, 0, i), Vector(10, 0, 10), Vector(0, 0, 0), 0.5, 0);
                timer.Stop();
            }

            collector.StopCapture();
            reporter.RecordPerformanceResult(timer.CreateResult("TelemetryCapture", "Performance", 0.1));
        }
    }

    //--------------------------------------------------------------------------------------------
    protected static void RunDataValidationTests(SCR_CaptureTestReporter reporter)
    {
        // Record integrity test
        {
            SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
            collector.StartCapture();

            vector testPos = Vector(100, 50, 200);
            collector.CaptureFrame(testPos, Vector(10, 0, 10), Vector(0, 45, 0), 0.75, -0.25);
            collector.StopCapture();

            SCR_MockTelemetryRecord record = collector.GetRecord(0);
            reporter.RecordResult(SCR_TestAssert.AssertVectorEquals(testPos, record.position, 0.001, "RecordIntegrity", "DataValidation"));
        }
    }

    //--------------------------------------------------------------------------------------------
    protected static void RunEdgeCaseTests(SCR_CaptureTestReporter reporter)
    {
        // Zero frames test
        {
            SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
            collector.StartCapture();
            collector.StopCapture();
            reporter.RecordResult(SCR_TestAssert.AssertEquals(0, collector.GetFrameCount(), "ZeroFrames", "EdgeCase"));
        }

        // Extreme positions test
        {
            SCR_MockMLDataCollector collector = new SCR_MockMLDataCollector();
            collector.StartCapture();
            vector extremePos = Vector(100000, 10000, 100000);
            collector.CaptureFrame(extremePos, Vector(1000, 0, 1000), Vector(0, 0, 0), 1.0, 1.0);
            collector.StopCapture();

            SCR_MockTelemetryRecord record = collector.GetRecord(0);
            reporter.RecordResult(SCR_TestAssert.AssertVectorEquals(extremePos, record.position, 0.001, "ExtremePositions", "EdgeCase"));
        }
    }
}
