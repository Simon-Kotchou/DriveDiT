// ============================================================================
// SCR_MultiCameraRig - Multi-Camera Capture System for World Model Training
// ============================================================================
//
// GAIA-2 style multi-camera rig with full intrinsic/extrinsic calibration.
// Captures synchronized camera data from multiple viewpoints mounted on vehicle.
//
// CAMERA INTRINSICS (K matrix):
//   K = [[fx,  0, cx],
//        [ 0, fy, cy],
//        [ 0,  0,  1]]
//   where:
//     fy = H / (2 * tan(FOV_vertical / 2))
//     fx = fy (square pixels assumed)
//     cx = W / 2
//     cy = H / 2
//
// CAMERA EXTRINSICS ([R|t] matrix):
//   T_world_camera = T_world_vehicle * T_vehicle_camera
//   where T_vehicle_camera is the mount offset (position + rotation)
//
// OUTPUT FORMAT (per frame):
//   - Frame timestamp and ID
//   - Per-camera: K matrix (3x3), [R|t] matrix (3x4), image dimensions
//   - Vehicle pose (position + orientation)
//
// ============================================================================

// ============================================================================
// Camera Mount Position Enum
// ============================================================================
enum SCR_CameraMountPosition
{
    FRONT = 0,       // Front-facing camera
    FRONT_LEFT,      // Front-left 45 degrees
    FRONT_RIGHT,     // Front-right 45 degrees
    LEFT,            // Side-view left 90 degrees
    RIGHT,           // Side-view right 90 degrees
    REAR,            // Rear-facing camera
    REAR_LEFT,       // Rear-left 135 degrees
    REAR_RIGHT,      // Rear-right 135 degrees
    COUNT
}

// ============================================================================
// Camera Configuration Structure
// ============================================================================
class SCR_CameraConfig
{
    // Camera identification
    int m_iCameraIndex;
    SCR_CameraMountPosition m_eMountPosition;
    string m_sCameraName;

    // Image dimensions
    int m_iImageWidth;
    int m_iImageHeight;

    // Field of view (vertical, in degrees)
    float m_fVerticalFOV;

    // Near/far clipping planes
    float m_fNearPlane;
    float m_fFarPlane;

    // Mount offset relative to vehicle center
    // Position: offset from vehicle origin (meters)
    vector m_vMountPosition;

    // Mount rotation: yaw, pitch, roll (degrees)
    // Yaw: rotation around vertical axis (0 = forward, 90 = left, -90 = right, 180 = rear)
    // Pitch: rotation around lateral axis (positive = looking up)
    // Roll: rotation around longitudinal axis
    vector m_vMountRotation;

    // Enable/disable this camera
    bool m_bEnabled;

    // ========================================================================
    // Constructor with defaults
    // ========================================================================
    void SCR_CameraConfig()
    {
        m_iCameraIndex = 0;
        m_eMountPosition = SCR_CameraMountPosition.FRONT;
        m_sCameraName = "Camera_0";
        m_iImageWidth = 1920;
        m_iImageHeight = 1080;
        m_fVerticalFOV = 60.0;
        m_fNearPlane = 0.1;
        m_fFarPlane = 1000.0;
        m_vMountPosition = "0 1.5 0";  // 1.5m above vehicle origin
        m_vMountRotation = "0 0 0";    // Forward-facing
        m_bEnabled = true;
    }

    // ========================================================================
    // Create camera with specific mount position preset
    // ========================================================================
    static SCR_CameraConfig CreateFromMountPosition(SCR_CameraMountPosition mountPos, int index)
    {
        SCR_CameraConfig config = new SCR_CameraConfig();
        config.m_iCameraIndex = index;
        config.m_eMountPosition = mountPos;
        config.m_bEnabled = true;

        // Standard vehicle camera height and offset
        float cameraHeight = 1.5;      // 1.5m above vehicle center
        float forwardOffset = 1.0;     // 1m forward of center for front cameras
        float rearOffset = -1.5;       // 1.5m behind center for rear cameras
        float lateralOffset = 0.9;     // 0.9m lateral offset for side views

        switch (mountPos)
        {
            case SCR_CameraMountPosition.FRONT:
                config.m_sCameraName = "Camera_Front";
                config.m_vMountPosition = Vector(0, cameraHeight, forwardOffset);
                config.m_vMountRotation = "0 0 0";  // Forward
                config.m_fVerticalFOV = 60.0;
                break;

            case SCR_CameraMountPosition.FRONT_LEFT:
                config.m_sCameraName = "Camera_FrontLeft";
                config.m_vMountPosition = Vector(-lateralOffset, cameraHeight, forwardOffset * 0.5);
                config.m_vMountRotation = "45 0 0";  // 45 degrees left
                config.m_fVerticalFOV = 70.0;
                break;

            case SCR_CameraMountPosition.FRONT_RIGHT:
                config.m_sCameraName = "Camera_FrontRight";
                config.m_vMountPosition = Vector(lateralOffset, cameraHeight, forwardOffset * 0.5);
                config.m_vMountRotation = "-45 0 0";  // 45 degrees right
                config.m_fVerticalFOV = 70.0;
                break;

            case SCR_CameraMountPosition.LEFT:
                config.m_sCameraName = "Camera_Left";
                config.m_vMountPosition = Vector(-lateralOffset, cameraHeight, 0);
                config.m_vMountRotation = "90 0 0";  // 90 degrees left
                config.m_fVerticalFOV = 75.0;
                break;

            case SCR_CameraMountPosition.RIGHT:
                config.m_sCameraName = "Camera_Right";
                config.m_vMountPosition = Vector(lateralOffset, cameraHeight, 0);
                config.m_vMountRotation = "-90 0 0";  // 90 degrees right
                config.m_fVerticalFOV = 75.0;
                break;

            case SCR_CameraMountPosition.REAR:
                config.m_sCameraName = "Camera_Rear";
                config.m_vMountPosition = Vector(0, cameraHeight, rearOffset);
                config.m_vMountRotation = "180 0 0";  // 180 degrees (facing backward)
                config.m_fVerticalFOV = 70.0;
                break;

            case SCR_CameraMountPosition.REAR_LEFT:
                config.m_sCameraName = "Camera_RearLeft";
                config.m_vMountPosition = Vector(-lateralOffset, cameraHeight, rearOffset * 0.5);
                config.m_vMountRotation = "135 0 0";  // 135 degrees left
                config.m_fVerticalFOV = 70.0;
                break;

            case SCR_CameraMountPosition.REAR_RIGHT:
                config.m_sCameraName = "Camera_RearRight";
                config.m_vMountPosition = Vector(lateralOffset, cameraHeight, rearOffset * 0.5);
                config.m_vMountRotation = "-135 0 0";  // 135 degrees right
                config.m_fVerticalFOV = 70.0;
                break;
        }

        return config;
    }

    // ========================================================================
    // Get aspect ratio
    // ========================================================================
    float GetAspectRatio()
    {
        if (m_iImageHeight <= 0)
            return 1.0;
        return m_iImageWidth / m_iImageHeight;
    }
}

// ============================================================================
// Camera Intrinsic Matrix
// ============================================================================
class SCR_CameraIntrinsics
{
    // Focal lengths in pixels
    float fx;
    float fy;

    // Principal point (optical center)
    float cx;
    float cy;

    // Image dimensions
    int width;
    int height;

    // Source FOV (for reference)
    float verticalFOV;

    // ========================================================================
    // Compute intrinsics from FOV and image dimensions
    // ========================================================================
    void ComputeFromFOV(float fovVerticalDegrees, int imageWidth, int imageHeight)
    {
        width = imageWidth;
        height = imageHeight;
        verticalFOV = fovVerticalDegrees;

        // Convert FOV to radians
        float fovRad = fovVerticalDegrees * Math.DEG2RAD;

        // Compute focal length: fy = H / (2 * tan(FOV_v / 2))
        float halfFovRad = fovRad * 0.5;
        float tanHalfFov = Math.Tan(halfFovRad);

        if (tanHalfFov > 0.0001)
        {
            fy = height / (2.0 * tanHalfFov);
        }
        else
        {
            fy = height * 10.0;  // Very narrow FOV fallback
        }

        // Square pixels assumption: fx = fy
        fx = fy;

        // Principal point at image center
        cx = width * 0.5;
        cy = height * 0.5;
    }

    // ========================================================================
    // Get K matrix as array [fx, 0, cx, 0, fy, cy, 0, 0, 1]
    // Row-major order for 3x3 matrix
    // ========================================================================
    void GetKMatrix(out float kMatrix[9])
    {
        // Row 0
        kMatrix[0] = fx;
        kMatrix[1] = 0.0;
        kMatrix[2] = cx;

        // Row 1
        kMatrix[3] = 0.0;
        kMatrix[4] = fy;
        kMatrix[5] = cy;

        // Row 2
        kMatrix[6] = 0.0;
        kMatrix[7] = 0.0;
        kMatrix[8] = 1.0;
    }

    // ========================================================================
    // Format K matrix as string for output
    // ========================================================================
    string ToCSVString()
    {
        // Output: fx,fy,cx,cy,width,height
        string result = fx.ToString(12, 6) + ",";
        result += fy.ToString(12, 6) + ",";
        result += cx.ToString(10, 4) + ",";
        result += cy.ToString(10, 4) + ",";
        result += width.ToString() + ",";
        result += height.ToString();
        return result;
    }
}

// ============================================================================
// Camera Extrinsic Matrix (Pose)
// ============================================================================
class SCR_CameraExtrinsics
{
    // Rotation matrix (3x3) in row-major order
    float R[9];

    // Translation vector (3x1)
    float t[3];

    // Combined [R|t] matrix (3x4) in row-major order
    float Rt[12];

    // World position for reference
    vector worldPosition;

    // World rotation (yaw, pitch, roll) for reference
    vector worldRotation;

    // ========================================================================
    // Compute extrinsics from vehicle transform and camera mount offset
    // ========================================================================
    void ComputeFromVehicleAndMount(vector vehicleTransform[4], vector mountPosition, vector mountRotationDeg)
    {
        // vehicleTransform[0] = right (X)
        // vehicleTransform[1] = up (Y)
        // vehicleTransform[2] = forward (Z)
        // vehicleTransform[3] = position

        // Step 1: Build camera mount transform in vehicle space
        vector mountTransform[4];
        BuildMountTransform(mountPosition, mountRotationDeg, mountTransform);

        // Step 2: Compute world transform: T_world_camera = T_world_vehicle * T_vehicle_camera
        vector cameraWorldTransform[4];
        Math3D.MatrixMultiply4(vehicleTransform, mountTransform, cameraWorldTransform);

        // Step 3: Extract rotation matrix (3x3) from world transform
        // The rotation part is the upper-left 3x3 of the 4x4 transform
        // cameraWorldTransform[0] = right axis
        // cameraWorldTransform[1] = up axis
        // cameraWorldTransform[2] = forward axis

        // For camera extrinsics, we need R that transforms world points to camera space
        // This is the inverse (transpose for orthonormal) of the camera's world rotation

        // Camera axes in world space
        vector camRight = cameraWorldTransform[0];
        vector camUp = cameraWorldTransform[1];
        vector camForward = cameraWorldTransform[2];
        vector camPosition = cameraWorldTransform[3];

        // Store world position for reference
        worldPosition = camPosition;

        // Compute world rotation (approximate yaw/pitch/roll)
        ComputeYawPitchRoll(camForward, camUp, worldRotation);

        // R matrix: transforms world points to camera frame
        // R = [camRight, camUp, camForward]^T (each row is an axis)
        // Row 0: right axis (X in camera space)
        R[0] = camRight[0];
        R[1] = camRight[1];
        R[2] = camRight[2];

        // Row 1: up axis (Y in camera space)
        R[3] = camUp[0];
        R[4] = camUp[1];
        R[5] = camUp[2];

        // Row 2: forward axis (Z in camera space / optical axis)
        R[6] = camForward[0];
        R[7] = camForward[1];
        R[8] = camForward[2];

        // t vector: -R * camera_position
        // This is the camera position expressed in camera coordinates
        t[0] = -(R[0] * camPosition[0] + R[1] * camPosition[1] + R[2] * camPosition[2]);
        t[1] = -(R[3] * camPosition[0] + R[4] * camPosition[1] + R[5] * camPosition[2]);
        t[2] = -(R[6] * camPosition[0] + R[7] * camPosition[1] + R[8] * camPosition[2]);

        // Build combined [R|t] matrix (3x4)
        // Row 0
        Rt[0] = R[0];
        Rt[1] = R[1];
        Rt[2] = R[2];
        Rt[3] = t[0];

        // Row 1
        Rt[4] = R[3];
        Rt[5] = R[4];
        Rt[6] = R[5];
        Rt[7] = t[1];

        // Row 2
        Rt[8] = R[6];
        Rt[9] = R[7];
        Rt[10] = R[8];
        Rt[11] = t[2];
    }

    // ========================================================================
    // Build mount transform matrix from position and rotation
    // ========================================================================
    protected void BuildMountTransform(vector position, vector rotationDeg, out vector transform[4])
    {
        // Convert rotation from degrees to radians
        float yawRad = rotationDeg[0] * Math.DEG2RAD;
        float pitchRad = rotationDeg[1] * Math.DEG2RAD;
        float rollRad = rotationDeg[2] * Math.DEG2RAD;

        // Pre-compute sin/cos
        float cy = Math.Cos(yawRad);
        float sy = Math.Sin(yawRad);
        float cp = Math.Cos(pitchRad);
        float sp = Math.Sin(pitchRad);
        float cr = Math.Cos(rollRad);
        float sr = Math.Sin(rollRad);

        // Build rotation matrix using ZYX (yaw-pitch-roll) convention
        // R = Rz(yaw) * Ry(pitch) * Rx(roll)

        // Right axis (X)
        transform[0] = Vector(
            cy * cp,
            sy * cp,
            -sp
        );

        // Up axis (Y)
        transform[1] = Vector(
            cy * sp * sr - sy * cr,
            sy * sp * sr + cy * cr,
            cp * sr
        );

        // Forward axis (Z)
        transform[2] = Vector(
            cy * sp * cr + sy * sr,
            sy * sp * cr - cy * sr,
            cp * cr
        );

        // Position
        transform[3] = position;
    }

    // ========================================================================
    // Compute yaw/pitch/roll from forward and up vectors
    // ========================================================================
    protected void ComputeYawPitchRoll(vector forward, vector up, out vector ypr)
    {
        // Yaw: angle in XZ plane
        float yaw = Math.Atan2(forward[0], forward[2]) * Math.RAD2DEG;

        // Pitch: angle from horizontal
        float pitch = Math.Asin(-forward[1]) * Math.RAD2DEG;

        // Roll: rotation around forward axis (simplified)
        float roll = Math.Atan2(up[0], up[1]) * Math.RAD2DEG;

        ypr = Vector(yaw, pitch, roll);
    }

    // ========================================================================
    // Format extrinsics as CSV string
    // ========================================================================
    string ToCSVString()
    {
        // Output: R11,R12,R13,R21,R22,R23,R31,R32,R33,t1,t2,t3
        string result = "";

        // Rotation matrix (row-major)
        for (int i = 0; i < 9; i++)
        {
            result += R[i].ToString(10, 8);
            result += ",";
        }

        // Translation vector
        result += t[0].ToString(10, 6) + ",";
        result += t[1].ToString(10, 6) + ",";
        result += t[2].ToString(10, 6);

        return result;
    }

    // ========================================================================
    // Get world position as CSV
    // ========================================================================
    string GetWorldPositionCSV()
    {
        return worldPosition[0].ToString(10, 4) + "," +
               worldPosition[1].ToString(10, 4) + "," +
               worldPosition[2].ToString(10, 4);
    }

    // ========================================================================
    // Get world rotation (YPR) as CSV
    // ========================================================================
    string GetWorldRotationCSV()
    {
        return worldRotation[0].ToString(8, 3) + "," +
               worldRotation[1].ToString(8, 3) + "," +
               worldRotation[2].ToString(8, 3);
    }
}

// ============================================================================
// Per-Frame Camera Data
// ============================================================================
class SCR_CameraFrameData
{
    int cameraIndex;
    string cameraName;
    ref SCR_CameraIntrinsics intrinsics;
    ref SCR_CameraExtrinsics extrinsics;

    void SCR_CameraFrameData()
    {
        intrinsics = new SCR_CameraIntrinsics();
        extrinsics = new SCR_CameraExtrinsics();
    }
}

// ============================================================================
// Multi-Camera Rig Component
// ============================================================================
[ComponentEditorProps(category: "GameScripted/DataCapture", description: "Multi-Camera Rig - GAIA-2 style synchronized multi-viewpoint capture")]
class SCR_MultiCameraRigClass: ScriptComponentClass
{
}

class SCR_MultiCameraRig: ScriptComponent
{
    // === CONFIGURATION ===
    [Attribute("1", UIWidgets.CheckBox, "Enable camera capture")]
    protected bool m_bEnableCapture;

    [Attribute("200", UIWidgets.Slider, "Capture interval in milliseconds", "50 1000 50")]
    protected int m_iCaptureIntervalMs;

    [Attribute("1920", UIWidgets.Slider, "Image width (pixels)", "640 3840 64")]
    protected int m_iImageWidth;

    [Attribute("1080", UIWidgets.Slider, "Image height (pixels)", "480 2160 48")]
    protected int m_iImageHeight;

    [Attribute("60", UIWidgets.Slider, "Default vertical FOV (degrees)", "30 120 5")]
    protected float m_fDefaultFOV;

    [Attribute("1", UIWidgets.CheckBox, "Enable front camera")]
    protected bool m_bEnableFront;

    [Attribute("1", UIWidgets.CheckBox, "Enable front-left camera")]
    protected bool m_bEnableFrontLeft;

    [Attribute("1", UIWidgets.CheckBox, "Enable front-right camera")]
    protected bool m_bEnableFrontRight;

    [Attribute("0", UIWidgets.CheckBox, "Enable rear camera")]
    protected bool m_bEnableRear;

    [Attribute("1.5", UIWidgets.Slider, "Camera mount height above vehicle center", "0.5 3.0 0.1")]
    protected float m_fMountHeight;

    [Attribute("1.0", UIWidgets.Slider, "Camera forward offset from vehicle center", "0.0 3.0 0.1")]
    protected float m_fForwardOffset;

    [Attribute("0.9", UIWidgets.Slider, "Camera lateral offset from vehicle center", "0.0 2.0 0.1")]
    protected float m_fLateralOffset;

    [Attribute("1", UIWidgets.CheckBox, "Verbose logging")]
    protected bool m_bVerboseLogging;

    // === CAMERA STATE ===
    protected ref array<ref SCR_CameraConfig> m_aCameraConfigs;
    protected ref array<ref SCR_CameraFrameData> m_aFrameData;

    // === SESSION STATE ===
    protected string m_sSessionPath;
    protected string m_sCameraDataPath;
    protected int m_iFrameCounter;
    protected bool m_bSessionInitialized;
    protected float m_fSessionStartTime;
    protected float m_fLastCaptureTime;

    // === VEHICLE REFERENCE ===
    protected IEntity m_VehicleEntity;

    // ========================================================================
    // Initialization
    // ========================================================================
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        // Initialize arrays
        m_aCameraConfigs = new array<ref SCR_CameraConfig>();
        m_aFrameData = new array<ref SCR_CameraFrameData>();

        m_bSessionInitialized = false;
        m_iFrameCounter = 0;

        // Store vehicle reference
        m_VehicleEntity = owner;

        // Setup cameras based on configuration
        SetupCameras();

        Print("[MultiCameraRig] Component initialized with " + m_aCameraConfigs.Count() + " cameras", LogLevel.NORMAL);
    }

    // ========================================================================
    // Setup camera configurations based on component attributes
    // ========================================================================
    protected void SetupCameras()
    {
        m_aCameraConfigs.Clear();
        m_aFrameData.Clear();

        int cameraIndex = 0;

        // Front camera
        if (m_bEnableFront)
        {
            SCR_CameraConfig config = SCR_CameraConfig.CreateFromMountPosition(SCR_CameraMountPosition.FRONT, cameraIndex);
            ConfigureCameraFromAttributes(config);
            m_aCameraConfigs.Insert(config);
            m_aFrameData.Insert(new SCR_CameraFrameData());
            cameraIndex++;
        }

        // Front-left camera
        if (m_bEnableFrontLeft)
        {
            SCR_CameraConfig config = SCR_CameraConfig.CreateFromMountPosition(SCR_CameraMountPosition.FRONT_LEFT, cameraIndex);
            ConfigureCameraFromAttributes(config);
            m_aCameraConfigs.Insert(config);
            m_aFrameData.Insert(new SCR_CameraFrameData());
            cameraIndex++;
        }

        // Front-right camera
        if (m_bEnableFrontRight)
        {
            SCR_CameraConfig config = SCR_CameraConfig.CreateFromMountPosition(SCR_CameraMountPosition.FRONT_RIGHT, cameraIndex);
            ConfigureCameraFromAttributes(config);
            m_aCameraConfigs.Insert(config);
            m_aFrameData.Insert(new SCR_CameraFrameData());
            cameraIndex++;
        }

        // Rear camera
        if (m_bEnableRear)
        {
            SCR_CameraConfig config = SCR_CameraConfig.CreateFromMountPosition(SCR_CameraMountPosition.REAR, cameraIndex);
            ConfigureCameraFromAttributes(config);
            m_aCameraConfigs.Insert(config);
            m_aFrameData.Insert(new SCR_CameraFrameData());
            cameraIndex++;
        }

        // Pre-compute intrinsics for all cameras (static per session)
        for (int i = 0; i < m_aCameraConfigs.Count(); i++)
        {
            SCR_CameraConfig cfg = m_aCameraConfigs[i];
            SCR_CameraFrameData frameData = m_aFrameData[i];

            frameData.cameraIndex = cfg.m_iCameraIndex;
            frameData.cameraName = cfg.m_sCameraName;
            frameData.intrinsics.ComputeFromFOV(cfg.m_fVerticalFOV, cfg.m_iImageWidth, cfg.m_iImageHeight);
        }
    }

    // ========================================================================
    // Apply component attributes to camera config
    // ========================================================================
    protected void ConfigureCameraFromAttributes(SCR_CameraConfig config)
    {
        config.m_iImageWidth = m_iImageWidth;
        config.m_iImageHeight = m_iImageHeight;

        // Adjust mount position based on attributes
        vector mountPos = config.m_vMountPosition;
        mountPos[1] = m_fMountHeight;

        // Scale forward/lateral offsets
        if (config.m_eMountPosition == SCR_CameraMountPosition.FRONT)
        {
            mountPos[2] = m_fForwardOffset;
        }
        else if (config.m_eMountPosition == SCR_CameraMountPosition.FRONT_LEFT ||
                 config.m_eMountPosition == SCR_CameraMountPosition.FRONT_RIGHT)
        {
            mountPos[2] = m_fForwardOffset * 0.5;
            if (mountPos[0] < 0)
                mountPos[0] = -m_fLateralOffset;
            else
                mountPos[0] = m_fLateralOffset;
        }

        config.m_vMountPosition = mountPos;
    }

    // ========================================================================
    // Initialize capture session
    // ========================================================================
    bool InitializeSession()
    {
        if (!m_bEnableCapture)
        {
            Print("[MultiCameraRig] Capture disabled", LogLevel.WARNING);
            return false;
        }

        if (m_bSessionInitialized)
        {
            Print("[MultiCameraRig] Session already initialized", LogLevel.WARNING);
            return true;
        }

        // Create base directory
        string basePath = "$profile:CameraData";
        FileIO.MakeDirectory(basePath);

        // Generate session ID from world time
        float worldTime = GetGame().GetWorld().GetWorldTime();
        int sessionID = worldTime;

        m_sSessionPath = basePath + "/session_" + sessionID.ToString();
        FileIO.MakeDirectory(m_sSessionPath);

        // Initialize camera data CSV
        m_sCameraDataPath = m_sSessionPath + "/camera_data.csv";

        if (!WriteCameraDataHeader())
        {
            Print("[MultiCameraRig] ERROR: Failed to create camera data file", LogLevel.ERROR);
            return false;
        }

        // Write camera configuration file
        WriteCameraConfigFile();

        m_fSessionStartTime = worldTime;
        m_bSessionInitialized = true;
        m_iFrameCounter = 0;
        m_fLastCaptureTime = 0;

        Print("[MultiCameraRig] Session initialized: " + m_sSessionPath, LogLevel.NORMAL);
        Print("[MultiCameraRig] Cameras: " + m_aCameraConfigs.Count() + ", Rate: " + (1000.0 / m_iCaptureIntervalMs).ToString(4, 1) + " Hz", LogLevel.NORMAL);

        return true;
    }

    // ========================================================================
    // Write CSV header for camera data
    // ========================================================================
    protected bool WriteCameraDataHeader()
    {
        FileHandle file = FileIO.OpenFile(m_sCameraDataPath, FileMode.WRITE);
        if (!file)
            return false;

        // Header format:
        // frame_id,timestamp_ms,camera_index,camera_name,
        // fx,fy,cx,cy,width,height,
        // R11,R12,R13,R21,R22,R23,R31,R32,R33,t1,t2,t3,
        // world_pos_x,world_pos_y,world_pos_z,
        // world_yaw,world_pitch,world_roll

        string header = "frame_id,timestamp_ms,camera_index,camera_name,";
        header += "fx,fy,cx,cy,width,height,";
        header += "R11,R12,R13,R21,R22,R23,R31,R32,R33,t1,t2,t3,";
        header += "world_pos_x,world_pos_y,world_pos_z,";
        header += "world_yaw,world_pitch,world_roll";

        file.WriteLine(header);
        file.Close();

        return true;
    }

    // ========================================================================
    // Write camera configuration file (static per session)
    // ========================================================================
    protected void WriteCameraConfigFile()
    {
        string configPath = m_sSessionPath + "/camera_config.txt";
        FileHandle file = FileIO.OpenFile(configPath, FileMode.WRITE);
        if (!file)
            return;

        file.WriteLine("=== MULTI-CAMERA RIG CONFIGURATION ===");
        file.WriteLine("");
        file.WriteLine("capture_interval_ms=" + m_iCaptureIntervalMs.ToString());
        file.WriteLine("capture_hz=" + (1000.0 / m_iCaptureIntervalMs).ToString(4, 1));
        file.WriteLine("num_cameras=" + m_aCameraConfigs.Count().ToString());
        file.WriteLine("");

        for (int i = 0; i < m_aCameraConfigs.Count(); i++)
        {
            SCR_CameraConfig cfg = m_aCameraConfigs[i];
            SCR_CameraFrameData frameData = m_aFrameData[i];

            file.WriteLine("=== CAMERA " + i.ToString() + ": " + cfg.m_sCameraName + " ===");
            file.WriteLine("index=" + cfg.m_iCameraIndex.ToString());
            file.WriteLine("name=" + cfg.m_sCameraName);
            file.WriteLine("mount_position=" + cfg.m_eMountPosition.ToString());
            file.WriteLine("image_width=" + cfg.m_iImageWidth.ToString());
            file.WriteLine("image_height=" + cfg.m_iImageHeight.ToString());
            file.WriteLine("vertical_fov=" + cfg.m_fVerticalFOV.ToString(6, 2));
            file.WriteLine("near_plane=" + cfg.m_fNearPlane.ToString(6, 3));
            file.WriteLine("far_plane=" + cfg.m_fFarPlane.ToString(8, 1));
            file.WriteLine("mount_pos_xyz=" + cfg.m_vMountPosition[0].ToString(6, 3) + "," +
                                              cfg.m_vMountPosition[1].ToString(6, 3) + "," +
                                              cfg.m_vMountPosition[2].ToString(6, 3));
            file.WriteLine("mount_rot_ypr=" + cfg.m_vMountRotation[0].ToString(6, 2) + "," +
                                              cfg.m_vMountRotation[1].ToString(6, 2) + "," +
                                              cfg.m_vMountRotation[2].ToString(6, 2));

            // Intrinsic matrix
            file.WriteLine("fx=" + frameData.intrinsics.fx.ToString(12, 6));
            file.WriteLine("fy=" + frameData.intrinsics.fy.ToString(12, 6));
            file.WriteLine("cx=" + frameData.intrinsics.cx.ToString(10, 4));
            file.WriteLine("cy=" + frameData.intrinsics.cy.ToString(10, 4));
            file.WriteLine("");
        }

        file.Close();
        Print("[MultiCameraRig] Camera config written to: " + configPath, LogLevel.NORMAL);
    }

    // ========================================================================
    // Main capture function - call from game loop
    // ========================================================================
    void CaptureFrame()
    {
        if (!m_bSessionInitialized || !m_bEnableCapture)
            return;

        if (!m_VehicleEntity)
            return;

        float currentTime = GetGame().GetWorld().GetWorldTime();

        // Rate limiting
        if ((currentTime - m_fLastCaptureTime) < m_iCaptureIntervalMs)
            return;

        m_fLastCaptureTime = currentTime;

        // Get vehicle world transform
        vector vehicleTransform[4];
        m_VehicleEntity.GetWorldTransform(vehicleTransform);

        // Capture data for each camera
        for (int i = 0; i < m_aCameraConfigs.Count(); i++)
        {
            SCR_CameraConfig cfg = m_aCameraConfigs[i];
            SCR_CameraFrameData frameData = m_aFrameData[i];

            if (!cfg.m_bEnabled)
                continue;

            // Compute extrinsics for this frame
            frameData.extrinsics.ComputeFromVehicleAndMount(
                vehicleTransform,
                cfg.m_vMountPosition,
                cfg.m_vMountRotation
            );

            // Write camera data row
            WriteCameraDataRow(frameData, currentTime);
        }

        m_iFrameCounter++;

        // Progress logging
        if (m_bVerboseLogging && m_iFrameCounter % 100 == 0)
        {
            float elapsedMin = (currentTime - m_fSessionStartTime) / 60000.0;
            Print("[MultiCameraRig] Frame " + m_iFrameCounter.ToString() + " (" + elapsedMin.ToString(6, 1) + " min)", LogLevel.VERBOSE);
        }
    }

    // ========================================================================
    // Write single camera data row to CSV
    // ========================================================================
    protected void WriteCameraDataRow(SCR_CameraFrameData frameData, float timestampMs)
    {
        FileHandle file = FileIO.OpenFile(m_sCameraDataPath, FileMode.APPEND);
        if (!file)
            return;

        string row = m_iFrameCounter.ToString() + ",";
        row += timestampMs.ToString(12, 1) + ",";
        row += frameData.cameraIndex.ToString() + ",";
        row += frameData.cameraName + ",";

        // Intrinsics
        row += frameData.intrinsics.ToCSVString() + ",";

        // Extrinsics
        row += frameData.extrinsics.ToCSVString() + ",";

        // World position
        row += frameData.extrinsics.GetWorldPositionCSV() + ",";

        // World rotation
        row += frameData.extrinsics.GetWorldRotationCSV();

        file.WriteLine(row);
        file.Close();
    }

    // ========================================================================
    // Get all camera frame data for external processing
    // ========================================================================
    array<ref SCR_CameraFrameData> GetCameraFrameData()
    {
        return m_aFrameData;
    }

    // ========================================================================
    // Get camera configurations
    // ========================================================================
    array<ref SCR_CameraConfig> GetCameraConfigs()
    {
        return m_aCameraConfigs;
    }

    // ========================================================================
    // Add custom camera
    // ========================================================================
    int AddCamera(SCR_CameraConfig config)
    {
        if (m_bSessionInitialized)
        {
            Print("[MultiCameraRig] Cannot add cameras after session started", LogLevel.ERROR);
            return -1;
        }

        config.m_iCameraIndex = m_aCameraConfigs.Count();
        m_aCameraConfigs.Insert(config);

        SCR_CameraFrameData frameData = new SCR_CameraFrameData();
        frameData.cameraIndex = config.m_iCameraIndex;
        frameData.cameraName = config.m_sCameraName;
        frameData.intrinsics.ComputeFromFOV(config.m_fVerticalFOV, config.m_iImageWidth, config.m_iImageHeight);
        m_aFrameData.Insert(frameData);

        Print("[MultiCameraRig] Added camera: " + config.m_sCameraName, LogLevel.NORMAL);
        return config.m_iCameraIndex;
    }

    // ========================================================================
    // Set custom camera configuration
    // ========================================================================
    void ConfigureCustomCameraRig(array<ref SCR_CameraConfig> configs)
    {
        if (m_bSessionInitialized)
        {
            Print("[MultiCameraRig] Cannot reconfigure after session started", LogLevel.ERROR);
            return;
        }

        m_aCameraConfigs.Clear();
        m_aFrameData.Clear();

        for (int i = 0; i < configs.Count(); i++)
        {
            SCR_CameraConfig cfg = configs[i];
            cfg.m_iCameraIndex = i;
            m_aCameraConfigs.Insert(cfg);

            SCR_CameraFrameData frameData = new SCR_CameraFrameData();
            frameData.cameraIndex = i;
            frameData.cameraName = cfg.m_sCameraName;
            frameData.intrinsics.ComputeFromFOV(cfg.m_fVerticalFOV, cfg.m_iImageWidth, cfg.m_iImageHeight);
            m_aFrameData.Insert(frameData);
        }

        Print("[MultiCameraRig] Configured custom rig with " + m_aCameraConfigs.Count() + " cameras", LogLevel.NORMAL);
    }

    // ========================================================================
    // Get intrinsic matrix K for a camera (row-major 3x3)
    // ========================================================================
    bool GetIntrinsicMatrix(int cameraIndex, out float kMatrix[9])
    {
        if (cameraIndex < 0 || cameraIndex >= m_aFrameData.Count())
            return false;

        m_aFrameData[cameraIndex].intrinsics.GetKMatrix(kMatrix);
        return true;
    }

    // ========================================================================
    // Get extrinsic matrix [R|t] for a camera (row-major 3x4)
    // Must call CaptureFrame first to update extrinsics
    // ========================================================================
    bool GetExtrinsicMatrix(int cameraIndex, out float rtMatrix[12])
    {
        if (cameraIndex < 0 || cameraIndex >= m_aFrameData.Count())
            return false;

        for (int i = 0; i < 12; i++)
        {
            rtMatrix[i] = m_aFrameData[cameraIndex].extrinsics.Rt[i];
        }
        return true;
    }

    // ========================================================================
    // Project world point to image coordinates for a camera
    // ========================================================================
    bool ProjectWorldToImage(int cameraIndex, vector worldPoint, out float imageX, out float imageY)
    {
        if (cameraIndex < 0 || cameraIndex >= m_aFrameData.Count())
            return false;

        SCR_CameraFrameData data = m_aFrameData[cameraIndex];

        // Transform world point to camera coordinates: P_cam = R * P_world + t
        float px = data.extrinsics.R[0] * worldPoint[0] +
                   data.extrinsics.R[1] * worldPoint[1] +
                   data.extrinsics.R[2] * worldPoint[2] +
                   data.extrinsics.t[0];

        float py = data.extrinsics.R[3] * worldPoint[0] +
                   data.extrinsics.R[4] * worldPoint[1] +
                   data.extrinsics.R[5] * worldPoint[2] +
                   data.extrinsics.t[1];

        float pz = data.extrinsics.R[6] * worldPoint[0] +
                   data.extrinsics.R[7] * worldPoint[1] +
                   data.extrinsics.R[8] * worldPoint[2] +
                   data.extrinsics.t[2];

        // Check if point is behind camera
        if (pz <= 0)
            return false;

        // Project to image plane: p = K * P_cam (perspective division)
        imageX = data.intrinsics.fx * (px / pz) + data.intrinsics.cx;
        imageY = data.intrinsics.fy * (py / pz) + data.intrinsics.cy;

        // Check if within image bounds
        if (imageX < 0 || imageX >= data.intrinsics.width ||
            imageY < 0 || imageY >= data.intrinsics.height)
            return false;

        return true;
    }

    // ========================================================================
    // Session statistics
    // ========================================================================
    void GetSessionStats(out int frameCount, out float elapsedMinutes, out int cameraCount)
    {
        frameCount = m_iFrameCounter;
        elapsedMinutes = (GetGame().GetWorld().GetWorldTime() - m_fSessionStartTime) / 60000.0;
        cameraCount = m_aCameraConfigs.Count();
    }

    // ========================================================================
    // Finalize session
    // ========================================================================
    void FinalizeSession()
    {
        if (!m_bSessionInitialized)
            return;

        int frameCount;
        float elapsedMinutes;
        int cameraCount;
        GetSessionStats(frameCount, elapsedMinutes, cameraCount);

        // Write summary file
        string summaryPath = m_sSessionPath + "/summary.txt";
        FileHandle file = FileIO.OpenFile(summaryPath, FileMode.WRITE);
        if (file)
        {
            file.WriteLine("=== MULTI-CAMERA SESSION SUMMARY ===");
            file.WriteLine("total_frames=" + frameCount.ToString());
            file.WriteLine("duration_minutes=" + elapsedMinutes.ToString(8, 2));
            file.WriteLine("camera_count=" + cameraCount.ToString());
            file.WriteLine("total_camera_frames=" + (frameCount * cameraCount).ToString());
            file.WriteLine("capture_hz=" + (1000.0 / m_iCaptureIntervalMs).ToString(4, 1));
            file.Close();
        }

        Print("[MultiCameraRig] Session finalized: " + frameCount.ToString() + " frames, " +
              cameraCount.ToString() + " cameras, " + elapsedMinutes.ToString(6, 1) + " minutes", LogLevel.NORMAL);
        Print("[MultiCameraRig] Data saved to: " + m_sSessionPath, LogLevel.NORMAL);

        m_bSessionInitialized = false;
    }

    // ========================================================================
    // Public API
    // ========================================================================
    bool IsCapturing()
    {
        return m_bSessionInitialized && m_bEnableCapture;
    }

    string GetSessionPath()
    {
        return m_sSessionPath;
    }

    int GetFrameCount()
    {
        return m_iFrameCounter;
    }

    int GetCameraCount()
    {
        return m_aCameraConfigs.Count();
    }

    // ========================================================================
    // Cleanup
    // ========================================================================
    override void OnDelete(IEntity owner)
    {
        if (m_bSessionInitialized)
        {
            FinalizeSession();
        }

        super.OnDelete(owner);
    }
}
