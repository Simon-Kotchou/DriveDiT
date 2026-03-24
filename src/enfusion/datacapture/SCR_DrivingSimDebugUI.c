// ============================================================================
// SCR_DrivingSimDebugUI - Debug UI for AI Driving Simulator
// ============================================================================
//
// Provides real-time debug visualization and telemetry display for the
// AI Driving Simulator and ML Data Collection system.
//
// Registers in DiagMenu under AIScript category.
//
// ============================================================================

[ComponentEditorProps(category: "GameScripted/Debug", description: "Debug UI for AI Driving Simulator")]
class SCR_DrivingSimDebugUIClass: ScriptComponentClass
{
}

class SCR_DrivingSimDebugUI: ScriptComponent
{
    // References
    protected SCR_AIDrivingSimulator m_DrivingSimulator;
    protected SCR_MLDataCollector m_DataCollector;

    // Debug menu IDs (use values in a safe range)
    static const int MENU_DRIVING_SIM = 900;
    static const int MENU_SHOW_PANEL = 901;
    static const int MENU_SHOW_VEHICLE_DATA = 902;
    static const int MENU_SHOW_AI_INPUTS = 903;
    static const int MENU_SHOW_CAPTURE_STATUS = 904;

    // State
    protected bool m_bInitialized;
    protected float m_fStartTime;

    //------------------------------------------------------------------------
    override void OnPostInit(IEntity owner)
    {
        super.OnPostInit(owner);

        m_bInitialized = false;
        m_fStartTime = 0;

        // Register debug menus
        RegisterDebugMenus();

        // Find simulator components
        FindSimulatorComponents(owner);

        // Enable frame updates for debug UI
        SetEventMask(owner, EntityEvent.FRAME);

        Print("[DrivingSimDebugUI] Debug UI initialized", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void RegisterDebugMenus()
    {
        // Register main menu
        DiagMenu.RegisterMenu(MENU_DRIVING_SIM, "DrivingSim", "AI");

        // Register toggles
        DiagMenu.RegisterBool(MENU_SHOW_PANEL, "", "Show Telemetry Panel", "DrivingSim");
        DiagMenu.RegisterBool(MENU_SHOW_VEHICLE_DATA, "", "Show Vehicle Data", "DrivingSim");
        DiagMenu.RegisterBool(MENU_SHOW_AI_INPUTS, "", "Show AI Inputs", "DrivingSim");
        DiagMenu.RegisterBool(MENU_SHOW_CAPTURE_STATUS, "", "Show Capture Status", "DrivingSim");

        Print("[DrivingSimDebugUI] Registered debug menus", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    protected void FindSimulatorComponents(IEntity owner)
    {
        if (!owner)
            return;

        // Try to find on same entity
        m_DrivingSimulator = SCR_AIDrivingSimulator.Cast(owner.FindComponent(SCR_AIDrivingSimulator));
        m_DataCollector = SCR_MLDataCollector.Cast(owner.FindComponent(SCR_MLDataCollector));

        // If not found, try to find on game mode
        if (!m_DrivingSimulator || !m_DataCollector)
        {
            SCR_BaseGameMode gameMode = SCR_BaseGameMode.Cast(GetGame().GetGameMode());
            if (gameMode)
            {
                IEntity gameModeEntity = gameMode;
                if (!m_DrivingSimulator)
                    m_DrivingSimulator = SCR_AIDrivingSimulator.Cast(gameModeEntity.FindComponent(SCR_AIDrivingSimulator));
                if (!m_DataCollector)
                    m_DataCollector = SCR_MLDataCollector.Cast(gameModeEntity.FindComponent(SCR_MLDataCollector));
            }
        }

        if (m_DrivingSimulator)
            Print("[DrivingSimDebugUI] Found AIDrivingSimulator", LogLevel.NORMAL);
        if (m_DataCollector)
            Print("[DrivingSimDebugUI] Found MLDataCollector", LogLevel.NORMAL);
    }

    //------------------------------------------------------------------------
    override void EOnFrame(IEntity owner, float timeSlice)
    {
        super.EOnFrame(owner, timeSlice);

        // Only draw if panel is enabled
        if (!DiagMenu.GetBool(MENU_SHOW_PANEL))
            return;

        // Track start time
        if (m_fStartTime == 0)
            m_fStartTime = GetGame().GetWorld().GetWorldTime();

        // Draw debug UI
        DrawDebugPanel();
    }

    //------------------------------------------------------------------------
    protected void DrawDebugPanel()
    {
        DbgUI.Begin("AI Driving Simulator");

        float simTime = (GetGame().GetWorld().GetWorldTime() - m_fStartTime) / 1000.0;
        DbgUI.Text("=== DRIVING SIMULATOR DEBUG ===");
        DbgUI.Text("Simulation Time: " + simTime.ToString(8, 1) + "s");

        DbgUI.Text("");

        // Data capture status
        if (DiagMenu.GetBool(MENU_SHOW_CAPTURE_STATUS))
        {
            DrawCaptureStatus();
        }

        // Vehicle data (if we have access to simulation data)
        if (DiagMenu.GetBool(MENU_SHOW_VEHICLE_DATA))
        {
            DrawVehicleData();
        }

        // AI inputs
        if (DiagMenu.GetBool(MENU_SHOW_AI_INPUTS))
        {
            DrawAIInputs();
        }

        DbgUI.End();
    }

    //------------------------------------------------------------------------
    protected void DrawCaptureStatus()
    {
        DbgUI.Text("=== DATA CAPTURE ===");

        if (m_DataCollector)
        {
            if (m_DataCollector.IsCapturing())
            {
                DbgUI.Text("Status: CAPTURING");
                DbgUI.Text("Frames: " + m_DataCollector.GetFrameCount().ToString());
                DbgUI.Text("Session: " + m_DataCollector.GetSessionPath());
            }
            else
            {
                DbgUI.Text("Status: NOT ACTIVE");
            }
        }
        else
        {
            DbgUI.Text("Status: NO COLLECTOR");
        }

        DbgUI.Text("");
    }

    //------------------------------------------------------------------------
    protected void DrawVehicleData()
    {
        DbgUI.Text("=== VEHICLE DATA ===");

        // Find vehicles in the world by name pattern
        for (int i = 1; i <= 3; i++)
        {
            string vehicleName = "AI_Car_" + i.ToString();
            IEntity vehicle = GetGame().GetWorld().FindEntityByName(vehicleName);

            if (!vehicle)
                continue;

            VehicleWheeledSimulation sim = VehicleWheeledSimulation.Cast(vehicle.FindComponent(VehicleWheeledSimulation));
            if (!sim)
                continue;

            DbgUI.Text("--- " + vehicleName + " ---");

            float speed = sim.GetSpeedKmh();
            int gear = sim.GetGear();
            float rpm = sim.EngineGetRPM();
            bool engineOn = sim.EngineIsOn();

            string engineStatus = "OFF";
            if (engineOn)
                engineStatus = "ON";

            DbgUI.Text("Speed: " + speed.ToString(6, 1) + " km/h");
            DbgUI.Text("Gear: " + gear.ToString());
            DbgUI.Text("Engine: " + engineStatus + " (" + rpm.ToString(6, 0) + " RPM)");

            if (DiagMenu.GetBool(MENU_SHOW_AI_INPUTS))
            {
                float steering = sim.GetSteering();
                float throttle = sim.GetThrottle();
                float brake = sim.GetBrake();
                float clutch = sim.GetClutch();

                DbgUI.Text("Steering: " + steering.ToString(5, 3));
                DbgUI.Text("Throttle: " + throttle.ToString(5, 3));
                DbgUI.Text("Brake: " + brake.ToString(5, 3));
                DbgUI.Text("Clutch: " + clutch.ToString(5, 3));
            }

            DbgUI.Text("");
        }
    }

    //------------------------------------------------------------------------
    protected void DrawAIInputs()
    {
        // This is handled inline with vehicle data when the flag is set
        // Additional AI-specific info can go here
        DbgUI.Text("=== AI CONTROL INFO ===");
        DbgUI.Text("(AI inputs shown per vehicle above)");
        DbgUI.Text("");
    }

    //------------------------------------------------------------------------
    override void OnDelete(IEntity owner)
    {
        // Unregister menus
        DiagMenu.Unregister(MENU_SHOW_PANEL);
        DiagMenu.Unregister(MENU_SHOW_VEHICLE_DATA);
        DiagMenu.Unregister(MENU_SHOW_AI_INPUTS);
        DiagMenu.Unregister(MENU_SHOW_CAPTURE_STATUS);
        DiagMenu.Unregister(MENU_DRIVING_SIM);

        super.OnDelete(owner);
    }
}
