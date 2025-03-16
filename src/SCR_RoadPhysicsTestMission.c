// Simplified Road Physics Test Mission
[EntityEditorProps(category: "GameScripted/Missions", description: "Simple Physics Test Mission")]
class SCR_RoadPhysicsTestMissionClass: SCR_BaseGameModeClass
{
}

class SCR_RoadPhysicsTestMission: SCR_BaseGameMode
{
    [Attribute("$logs:PhysicsTest", UIWidgets.EditBox, "Output directory for recorded data")]
    private string m_OutputDirectory;
    
    [Attribute("", UIWidgets.ResourceNamePicker, "Vehicle prefab to test", params: "et")]
    private ResourceName m_VehiclePrefab;
    
    [Attribute(defvalue: "0 0 0", UIWidgets.EditBox, "Starting position for test (or use entity position if zero)")]
    protected vector m_vStartPosition;
    
    [Attribute(defvalue: "30", UIWidgets.Slider, "Target speed (km/h)", "5 200 5")]
    protected float m_fTargetSpeed;
    
    [Attribute(defvalue: "1", UIWidgets.CheckBox, "Record physics data")]
    protected bool m_bRecordPhysics;
    
    [Attribute(defvalue: "1", UIWidgets.CheckBox, "Follow vehicle with camera")]
    protected bool m_bFollowVehicle;
    
    // Internal variables
    private IEntity m_MainCamera;
    private Vehicle m_Vehicle;
    private ref array<IEntity> m_SpawnedEntities = {};
    
    // Physics recording variables
    private FileHandle m_PhysicsDataFile;
    private float m_TestDuration = 0;
    
    //------------------------------------------------------------------------------------------------
    override void OnGameStart()
    {
        super.OnGameStart();
        
        Print("SCR_RoadPhysicsTestMission: Initializing...");
        
        // Create output directory
        FileIO.MakeDirectory(m_OutputDirectory);
        
        // Give a delay to allow the world to fully load
        GetGame().GetCallqueue().CallLater(InitializeTest, 2000, false);
    }
    
    //------------------------------------------------------------------------------------------------
    void InitializeTest()
    {
        // Create camera first
        CreateCamera();
        
        // Spawn the test vehicle
        SpawnTestVehicle();
        
        // Initialize physics recording if enabled
        if (m_bRecordPhysics)
        {
            InitializePhysicsRecording();
        }
        
        // Start frame updates
        SetEventMask(EntityEvent.FRAME);
        
        Print("SCR_RoadPhysicsTestMission: Test initialized successfully");
    }
    
    //------------------------------------------------------------------------------------------------
    void CreateCamera()
    {
        // Create a script camera at origin
        EntitySpawnParams params = new EntitySpawnParams();
        params.TransformMode = ETransformMode.WORLD;
        Math3D.MatrixIdentity4(params.Transform);
        
        Resource cameraRes = Resource.Load("{917BDCEF4C2CD325}Prefabs/Cameras/Camera.et");
        m_MainCamera = GetGame().SpawnEntityPrefab(cameraRes, GetGame().GetWorld(), params);
        
        if (m_MainCamera)
        {
            m_MainCamera.SetName("PhysicsTestCamera");
            m_SpawnedEntities.Insert(m_MainCamera);
            Print("Camera created successfully");
        }
        else
        {
            Print("Failed to create camera", LogLevel.ERROR);
        }
    }
    
    //------------------------------------------------------------------------------------------------
    void SpawnTestVehicle()
    {
        // Determine spawn position
        vector spawnPosition = m_vStartPosition;
        if (spawnPosition == vector.Zero)
        {
            spawnPosition = GetOrigin();
        }
        
        // Ensure vehicle spawns above terrain
        float surfaceY = GetGame().GetWorld().GetSurfaceY(spawnPosition[0], spawnPosition[2]);
        spawnPosition[1] = surfaceY + 0.5;
        
        // Calculate default rotation (along world X axis)
        vector mat[4];
        Math3D.DirectionAndUpMatrix(vector.Forward, vector.Up, mat);
        
        // Create spawn parameters
        EntitySpawnParams params = new EntitySpawnParams();
        params.TransformMode = ETransformMode.WORLD;
        params.Transform = mat;
        params.Transform[3] = spawnPosition;
        
        // Spawn the vehicle
        IEntity vehicleEntity;
        
        if (m_VehiclePrefab.IsEmpty())
        {
            // Use a default vehicle if none specified
            Print("No vehicle prefab specified, using default UAZ");
            Resource defaultVehicle = Resource.Load("{5436629450D8387A}Prefabs/Vehicles/Wheeled/UAZ469/UAZ469.et");
            vehicleEntity = GetGame().SpawnEntityPrefab(defaultVehicle, GetGame().GetWorld(), params);
        }
        else
        {
            vehicleEntity = GetGame().SpawnEntityPrefab(Resource.Load(m_VehiclePrefab), GetGame().GetWorld(), params);
        }
        
        m_Vehicle = Vehicle.Cast(vehicleEntity);
        
        if (m_Vehicle)
        {
            m_Vehicle.SetName("TestVehicle");
            m_SpawnedEntities.Insert(m_Vehicle);
            Print("Vehicle spawned successfully at " + spawnPosition.ToString());
        }
        else
        {
            Print("Failed to spawn vehicle!", LogLevel.ERROR);
        }
    }
    
    //------------------------------------------------------------------------------------------------
    void InitializePhysicsRecording()
    {
        // Create physics data file
        string physicsFilename = m_OutputDirectory + "/physics_data.csv";
        m_PhysicsDataFile = FileIO.OpenFile(physicsFilename, FileMode.WRITE);
        
        if (m_PhysicsDataFile)
        {
            // Write CSV header
            m_PhysicsDataFile.WriteLine("Time,PosX,PosY,PosZ,SpeedKmh");
            Print("Physics data recording enabled: " + physicsFilename);
        }
        else
        {
            Print("Failed to create physics data file!", LogLevel.ERROR);
        }
    }
    
    //------------------------------------------------------------------------------------------------
    override void EOnFrame(IEntity owner, float timeSlice)
    {
        super.EOnFrame(owner, timeSlice);
        
        if (!m_Vehicle)
            return;
            
        // Update test duration
        m_TestDuration += timeSlice;
        
        // Update camera position if following vehicle
        if (m_bFollowVehicle && m_MainCamera)
        {
            UpdateCameraPosition();
        }
        
        // Record physics data
        if (m_bRecordPhysics && m_PhysicsDataFile)
        {
            RecordPhysicsData();
        }
    }
    
    //------------------------------------------------------------------------------------------------
    void UpdateCameraPosition()
    {
        ScriptCamera camera = ScriptCamera.Cast(m_MainCamera);
        if (!camera)
            return;
            
        vector vehiclePos = m_Vehicle.GetOrigin();
        
        // Get vehicle direction
        vector mat[4];
        m_Vehicle.GetTransform(mat);
        vector vehicleDir = mat[2]; // Forward direction
        vehicleDir[1] = 0; // Zero out vertical
        vehicleDir.Normalize();
        
        // Position camera behind and above vehicle
        vector cameraPos = vehiclePos;
        cameraPos[0] = cameraPos[0] - (vehicleDir[0] * 10.0); // 10m behind
        cameraPos[2] = cameraPos[2] - (vehicleDir[2] * 10.0);
        cameraPos[1] = cameraPos[1] + 3.0; // 3m above
        
        // Ensure camera doesn't go below terrain
        float terrainY = GetGame().GetWorld().GetSurfaceY(cameraPos[0], cameraPos[2]);
        if (cameraPos[1] < terrainY + 1.0)
        {
            cameraPos[1] = terrainY + 1.0;
        }
        
        camera.SetOrigin(cameraPos);
        
        // Point camera at vehicle
        vector lookDir = vehiclePos - cameraPos;
        lookDir.Normalize();
        
        camera.SetAngles(lookDir);
    }
    
    //------------------------------------------------------------------------------------------------
    void RecordPhysicsData()
    {
        if (!m_PhysicsDataFile || !m_Vehicle)
            return;
            
        Physics physics = m_Vehicle.GetPhysics();
        if (!physics)
            return;
            
        // Get current time
        float currentTime = GetGame().GetWorld().GetWorldTime();
        
        // Get vehicle data
        vector position = m_Vehicle.GetOrigin();
        vector velocity = physics.GetVelocity();
        float speed = velocity.Length() * Physics.MS2KMH; // Convert m/s to km/h
        
        // Format data as CSV line
        string line = string.Format("%.3f,%.3f,%.3f,%.3f,%.3f",
            currentTime,
            position[0], position[1], position[2],
            speed
        );
        
        m_PhysicsDataFile.WriteLine(line);
    }
    
    //------------------------------------------------------------------------------------------------
    void CloseDataFiles()
    {
        if (m_PhysicsDataFile)
        {
            m_PhysicsDataFile.Close();
            m_PhysicsDataFile = null;
        }
    }
    
    //------------------------------------------------------------------------------------------------
    override void OnGameEnd()
    {
        // Clean up resources
        CloseDataFiles();
        
        // Clean up spawned entities
        for (int i = 0; i < m_SpawnedEntities.Count(); i++)
        {
            IEntity entity = m_SpawnedEntities[i];
            if (entity)
            {
                delete entity;
            }
        }
        
        m_SpawnedEntities.Clear();
        
        super.OnGameEnd();
    }
}