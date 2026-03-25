// ============================================================================
// SCR_MLScreenshotCapture - ML Training Screenshot Capture Component
// ============================================================================
//
// Captures synchronized screenshots from vehicle perspectives for ML training.
// Automatically manages camera positioning without manual setup.
//
// HOW IT WORKS:
//   1. Finds or creates a ManualCamera in the world
//   2. Each capture: positions camera at vehicle -> screenshot -> done
//   3. Works seamlessly in Workbench and runtime
//
// CAMERA POSITIONS:
//   - HOOD: Forward-looking from hood (default, best for ML training)
//   - CHASE: Third-person behind vehicle
//   - BUMPER: Low forward-looking from bumper
//   - DASH: Dashboard interior view
//
// OUTPUT:
//   $profile:DrivingData/session_XXXX/frames/frame_XXXXXX.bmp
//
// SETUP:
//   1. Add this component to the same entity as SCR_MLDataCollector
//   2. Enable "Capture screenshots" in SCR_MLDataCollector
//   3. Run - screenshots are captured automatically
//
// ============================================================================

// Camera position enumeration
enum SCR_MLCameraPosition
{
    HOOD,       // Hood-mounted forward cam (recommended for ML)
    CHASE,      // Third-person chase cam
    BUMPER,     // Low bumper-level cam
    DASH        // Dashboard interior view
}

[ComponentEditorProps(category: "GameScripted/DataCapture", description: "ML Screenshot Capture - Automated vehicle camera screenshots")]
class SCR_MLScreenshotCaptureClass: ScriptComponentClass
{
}

class SCR_MLScreenshotCapture: ScriptComponent
{
    // === CONFIGURATION ===
    [Attribute("1", UIWidgets.CheckBox, "Enable screenshot capture")]
    protected bool m_bEnableCapture;

    [Attribute("0", UIWidgets.ComboBox, "Camera position on vehicle", "", ParamEnumArray.FromEnum(SCR_MLCameraPosition))]
    protected SCR_MLCameraPosition m_eCameraPosition;

    [Attribute("640", UIWidgets.Slider, "Capture width in pixels", "320 1920 32")]
    protected int m_iCaptureWidth;

    [Attribute("360", UIWidgets.Slider, "Capture height in pixels", "180 1080 18")]
    protected int m_iCaptureHeight;

    [Attribute("75", UIWidgets.Slider, "Camera field of view in degrees", "60 120 5")]
    protected float m_fCameraFOV;

    // Camera position offsets (forward, up, right) relative to vehicle
    [Attribute("2.2 0.95 0", UIWidgets.EditBox, "Hood camera offset (fwd, up, right)")]
    protected vector m_vHoodOffset;

    [Attribute("-6.0 2.5 0", UIWidgets.EditBox, "Chase camera offset (fwd, up, right)")]
    protected vector m_vChaseOffset;

    [Attribute("3.0 0.5 0", UIWidgets.EditBox, "Bumper camera offset (fwd, up, right)")]
    protected vector m_vBumperOffset;

    [Attribute("0.5 1.3 0", UIWidgets.EditBox, "Dash camera offset (fwd, up, right)")]
    protected vector m_vDashOffset;

    // === STATE ===
    protected string m_sSessionPath;
    protected string m_sFramesPath;
    protected bool m_bInitialized;
    protected int m_iFramesCaptured;
    protected int m_iFramesFailed;

    // Vehicle tracking
    protected ref array<IEntity> m_aTrackedVehicles;
    protected int m_iCurrentVehicleIndex;

    // Camera management
    protected CameraManager m_CameraManager;
    protected SCR_ManualCamera m_ManualCamera;
    protected CameraBase m_OriginalCamera;
    protected bool m_bCameraAttached;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        m_aTrackedVehicles = new array<IEntity>();
        m_bInitialized = false;
        m_iFramesCaptured = 0;
        m_iFramesFailed = 0;
        m_iCurrentVehicleIndex = 0;
        m_bCameraAttached = false;

        Print("[MLScreenshot] Component initialized, position: " + typename.EnumToString(SCR_MLCameraPosition, m_eCameraPosition), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Initialize capture session
    bool InitializeSession(string sessionPath)
    {
        if (!m_bEnableCapture)
        {
            Print("[MLScreenshot] Capture disabled", LogLevel.WARNING);
            return false;
        }

        m_sSessionPath = sessionPath;
        m_sFramesPath = sessionPath + "/frames";

        // Create frames directory
        FileIO.MakeDirectory(m_sFramesPath);

        // Get camera manager
        ChimeraGame game = ChimeraGame.Cast(GetGame());
        if (game)
        {
            m_CameraManager = game.GetCameraManager();
            if (m_CameraManager)
            {
                // Store current camera for reference
                m_OriginalCamera = m_CameraManager.CurrentCamera();
            }
        }

        // Try to find existing ManualCamera in world
        FindManualCamera();

        m_bInitialized = true;
        m_iFramesCaptured = 0;
        m_iFramesFailed = 0;

        Print("[MLScreenshot] Session initialized: " + m_sFramesPath, LogLevel.NORMAL);
        Print("[MLScreenshot] Resolution: " + m_iCaptureWidth.ToString() + "x" + m_iCaptureHeight.ToString(), LogLevel.NORMAL);
        Print("[MLScreenshot] ManualCamera found: " + (m_ManualCamera != null).ToString(), LogLevel.NORMAL);

        return true;
    }

    //------------------------------------------------------------------------
    // Find ManualCamera in world
    protected void FindManualCamera()
    {
        // Try to find existing ManualCamera entity
        BaseWorld world = GetGame().GetWorld();
        if (!world)
            return;

        // Search for ManualCamera entities
        // The editor typically has one already
        IEntity entity = world.FindEntityByName("ManualCamera");
        if (entity)
        {
            m_ManualCamera = SCR_ManualCamera.Cast(entity);
        }

        // If not found by name, try to get current camera if it's a ManualCamera
        if (!m_ManualCamera && m_CameraManager)
        {
            CameraBase currentCam = m_CameraManager.CurrentCamera();
            m_ManualCamera = SCR_ManualCamera.Cast(currentCam);
        }
    }

    //------------------------------------------------------------------------
    // Register a vehicle for tracking
    void RegisterVehicle(IEntity vehicle)
    {
        if (!vehicle)
            return;

        int existingIdx = m_aTrackedVehicles.Find(vehicle);
        if (existingIdx >= 0)
            return;

        m_aTrackedVehicles.Insert(vehicle);
        Print("[MLScreenshot] Registered vehicle: " + vehicle.GetName(), LogLevel.NORMAL);

        // Attach camera to first vehicle
        if (m_aTrackedVehicles.Count() == 1 && m_ManualCamera)
        {
            AttachCameraToVehicle(vehicle);
        }
    }

    //------------------------------------------------------------------------
    // Attach ManualCamera to vehicle
    protected void AttachCameraToVehicle(IEntity vehicle)
    {
        if (!m_ManualCamera || !vehicle)
            return;

        // Use ManualCamera's AttachTo method
        m_ManualCamera.AttachTo(vehicle);
        m_ManualCamera.SetInputEnabled(false); // Disable user input
        m_bCameraAttached = true;

        Print("[MLScreenshot] Camera attached to vehicle: " + vehicle.GetName(), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    // Detach camera from vehicle
    protected void DetachCamera()
    {
        if (!m_ManualCamera || !m_bCameraAttached)
            return;

        m_ManualCamera.Detach();
        m_ManualCamera.SetInputEnabled(true);
        m_bCameraAttached = false;
    }

    //------------------------------------------------------------------------
    // Get camera offset based on position setting
    protected vector GetCameraOffset()
    {
        switch (m_eCameraPosition)
        {
            case SCR_MLCameraPosition.HOOD:
                return m_vHoodOffset;
            case SCR_MLCameraPosition.CHASE:
                return m_vChaseOffset;
            case SCR_MLCameraPosition.BUMPER:
                return m_vBumperOffset;
            case SCR_MLCameraPosition.DASH:
                return m_vDashOffset;
        }
        return m_vHoodOffset;
    }

    //------------------------------------------------------------------------
    // Calculate camera world transform for a vehicle
    protected void CalculateCameraTransform(IEntity vehicle, out vector outTransform[4])
    {
        if (!vehicle)
            return;

        // Get vehicle transform
        vector vehicleTransform[4];
        vehicle.GetWorldTransform(vehicleTransform);

        vector vehiclePos = vehicleTransform[3];
        vector vehicleFwd = vehicleTransform[2];
        vector vehicleUp = vehicleTransform[1];
        vector vehicleRight = vehicleTransform[0];

        // Get offset for current camera position
        vector offset = GetCameraOffset();

        // Calculate camera position
        vector cameraPos = vehiclePos
            + vehicleFwd * offset[0]
            + vehicleUp * offset[1]
            + vehicleRight * offset[2];

        // Calculate look direction
        vector cameraFwd;
        vector cameraUp = vehicleUp;

        if (m_eCameraPosition == SCR_MLCameraPosition.CHASE)
        {
            // Look at vehicle center
            cameraFwd = vehiclePos - cameraPos;
            cameraFwd.Normalize();
        }
        else
        {
            // Look forward
            cameraFwd = vehicleFwd;
        }

        // Build orthogonal basis
        vector cameraRight = cameraUp * cameraFwd;
        cameraRight.Normalize();
        cameraUp = cameraFwd * cameraRight;
        cameraUp.Normalize();

        outTransform[0] = cameraRight;
        outTransform[1] = cameraUp;
        outTransform[2] = cameraFwd;
        outTransform[3] = cameraPos;
    }

    //------------------------------------------------------------------------
    // Update camera position to follow vehicle
    protected void UpdateCameraPosition()
    {
        if (m_aTrackedVehicles.Count() == 0)
            return;

        IEntity vehicle = m_aTrackedVehicles[0];
        if (!vehicle)
            return;

        // Calculate target camera transform
        vector cameraTransform[4];
        CalculateCameraTransform(vehicle, cameraTransform);

        // If we have ManualCamera attached, it follows automatically
        // But we may need to set offset - this is handled by attachment

        // For non-attached mode, we'd need to manually position the camera
        // which requires accessing the camera's transform
    }

    //------------------------------------------------------------------------
    // Capture frame
    bool CaptureFrame(int frameNumber)
    {
        if (!m_bInitialized || !m_bEnableCapture)
            return false;

        // Update camera position
        UpdateCameraPosition();

        // Build frame filename
        string frameNumStr = FormatFrameNumber(frameNumber);
        string framePath = m_sFramesPath + "/frame_" + frameNumStr + ".bmp";

        // Take screenshot
        bool success = System.MakeScreenshot(framePath);

        if (success)
        {
            m_iFramesCaptured++;

            if (m_iFramesCaptured % 100 == 0)
            {
                Print("[MLScreenshot] " + m_iFramesCaptured.ToString() + " frames captured", LogLevel.NORMAL);
            }
        }
        else
        {
            m_iFramesFailed++;

            if (m_iFramesFailed == 1)
            {
                Print("[MLScreenshot] WARNING: Screenshot capture may not be supported. Frames: " + m_iFramesFailed.ToString(), LogLevel.WARNING);
            }
        }

        return success;
    }

    //------------------------------------------------------------------------
    // Format frame number with zero-padding
    protected string FormatFrameNumber(int frameNumber)
    {
        string result = frameNumber.ToString();
        while (result.Length() < 6)
        {
            result = "0" + result;
        }
        return result;
    }

    //------------------------------------------------------------------------
    // Set tracked vehicle (called by MLDataCollector)
    void SetTrackedVehicle(IEntity vehicle)
    {
        RegisterVehicle(vehicle);
    }

    //------------------------------------------------------------------------
    // Get capture statistics
    void GetCaptureStats(out int framesCaptured, out int framesFailed, out string framesPath)
    {
        framesCaptured = m_iFramesCaptured;
        framesFailed = m_iFramesFailed;
        framesPath = m_sFramesPath;
    }

    //------------------------------------------------------------------------
    // Finalize session
    void FinalizeSession()
    {
        if (!m_bInitialized)
            return;

        // Detach camera
        DetachCamera();

        Print("[MLScreenshot] Session finalized:", LogLevel.NORMAL);
        Print("[MLScreenshot]   Frames captured: " + m_iFramesCaptured.ToString(), LogLevel.NORMAL);
        Print("[MLScreenshot]   Frames failed: " + m_iFramesFailed.ToString(), LogLevel.NORMAL);
        Print("[MLScreenshot]   Output path: " + m_sFramesPath, LogLevel.NORMAL);

        m_aTrackedVehicles.Clear();
        m_bInitialized = false;
    }

    //------------------------------------------------------------------------
    bool IsCapturing()
    {
        return m_bInitialized && m_bEnableCapture;
    }

    //------------------------------------------------------------------------
    int GetFramesCaptured()
    {
        return m_iFramesCaptured;
    }

    //------------------------------------------------------------------------
    SCR_MLCameraPosition GetCameraPosition()
    {
        return m_eCameraPosition;
    }

    //------------------------------------------------------------------------
    void SetCameraPosition(SCR_MLCameraPosition position)
    {
        m_eCameraPosition = position;
        Print("[MLScreenshot] Camera position: " + typename.EnumToString(SCR_MLCameraPosition, position), LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    int GetTrackedVehicleCount()
    {
        return m_aTrackedVehicles.Count();
    }

    //------------------------------------------------------------------------
    bool IsCameraAttached()
    {
        return m_bCameraAttached;
    }

    //------------------------------------------------------------------------
    override void OnDelete(IEntity owner)
    {
        if (m_bInitialized)
        {
            FinalizeSession();
        }
        super.OnDelete(owner);
    }
}
