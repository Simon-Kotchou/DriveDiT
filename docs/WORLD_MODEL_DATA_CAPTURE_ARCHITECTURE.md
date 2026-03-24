# World Model Data Capture Pipeline Architecture

## System Overview

This document specifies the architecture for a comprehensive data capture pipeline in the Enfusion game engine, designed to generate high-fidelity synthetic training data for world models. The architecture aligns with state-of-the-art approaches from GAIA-2, comma.ai, Self-Forcing++, and V-JEPA 2.

```
+-----------------------------------------------------------------------------------+
|                           ENFUSION DATA CAPTURE PIPELINE                          |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  +------------------------+     +------------------------+     +---------------+  |
|  |   SENSOR SUBSYSTEM     |     |   CONTEXT SUBSYSTEM    |     |  SYNC ENGINE  |  |
|  +------------------------+     +------------------------+     +---------------+  |
|  | - Multi-Camera Rig     |     | - Vehicle Dynamics     |     | - Frame Clock |  |
|  | - Depth Raycast Grid   |     | - Road Topology        |     | - Buffer Mgmt |  |
|  | - Semantic Raycaster   |     | - Scene Entities       |     | - I/O Queue   |  |
|  +------------------------+     | - Environment State    |     +---------------+  |
|            |                    +------------------------+            |           |
|            |                              |                           |           |
|            v                              v                           v           |
|  +------------------------------------------------------------------------+       |
|  |                        FRAME AGGREGATOR                                |       |
|  |  - Timestamp alignment (5Hz base clock)                                |       |
|  |  - Anchor frame designation (Self-Forcing++)                           |       |
|  |  - Multi-modal tensor packing                                          |       |
|  +------------------------------------------------------------------------+       |
|                                      |                                            |
|                                      v                                            |
|  +------------------------------------------------------------------------+       |
|  |                     BINARY SERIALIZER                                  |       |
|  |  - .enfcap container format                                            |       |
|  |  - Chunk-based storage with index tables                               |       |
|  |  - Compression: LZ4 for depth, raw for RGB                             |       |
|  +------------------------------------------------------------------------+       |
|                                      |                                            |
|                                      v                                            |
|  +------------------------------------------------------------------------+       |
|  |                       OUTPUT STREAMS                                   |       |
|  |  session_XXXX/                                                         |       |
|  |    frames.enfcap      (binary container)                               |       |
|  |    manifest.json      (session metadata)                               |       |
|  |    road_topology.bin  (cached road graph)                              |       |
|  +------------------------------------------------------------------------+       |
|                                                                                   |
+-----------------------------------------------------------------------------------+
```

---

## 1. Data Schema Specification

### 1.1 Frame Data Record (Aligned with GAIA-2/comma.ai)

Each captured frame contains a complete multi-modal observation at timestamp `t`.

```
FrameRecord {
    // === TEMPORAL IDENTITY ===
    frame_id:           uint64      // Monotonic frame counter
    timestamp_ms:       float64     // World time in milliseconds
    session_id:         uint32      // Capture session identifier
    is_anchor_frame:    bool        // Self-Forcing++ anchor designation

    // === EGO VEHICLE STATE (comma.ai format: 200ms intervals) ===
    ego_transform:      float32[16] // Full 4x4 world transform matrix
    ego_velocity:       float32[3]  // World-space velocity vector (m/s)
    ego_angular_vel:    float32[3]  // Angular velocity (rad/s)
    ego_acceleration:   float32[3]  // World-space acceleration (m/s^2)

    // === VEHICLE DYNAMICS (VehicleWheeledSimulation) ===
    speed_kmh:          float32     // Current speed
    steering:           float32     // Steering input [-1, 1]
    throttle:           float32     // Throttle input [0, 1]
    brake:              float32     // Brake input [0, 1]
    clutch:             float32     // Clutch state [0, 1]
    gear:               int8        // Current gear (-1=R, 0=N, 1-6=forward)
    engine_rpm:         float32     // Engine RPM
    engine_on:          bool        // Engine running state
    handbrake:          bool        // Handbrake engaged

    // === CONTROL CONDITIONING (GAIA-2 format) ===
    control_input:      float32[4]  // [steer, accel, goal_x, goal_y]
    future_waypoints:   float32[10][3] // Next 10 waypoints (x,y,z)
    target_speed_kmh:   float32     // Navigation target speed

    // === CAMERA INTRINSICS (per camera) ===
    cameras: [
        {
            camera_id:      uint8       // Camera index (0=front, 1-3=surround)
            fov_vertical:   float32     // Vertical FOV in degrees
            near_plane:     float32     // Near clip distance
            far_plane:      float32     // Far clip distance
            aspect_ratio:   float32     // Width/Height ratio
            resolution:     uint16[2]   // [width, height]
            extrinsic:      float32[16] // Camera-to-world transform
            intrinsic:      float32[9]  // 3x3 intrinsic matrix (K)
        }
    ]

    // === DEPTH MAP (raycast-derived) ===
    depth_map:          float16[H][W]   // Depth in meters (per camera)
    depth_valid_mask:   uint8[H][W]     // Validity mask (sky=0, valid=1)

    // === SCENE ENTITIES (3D bounding boxes - GAIA-2 format) ===
    entities: [
        {
            entity_id:      uint32      // Unique entity identifier
            class_id:       uint16      // Semantic class (vehicle, ped, etc.)
            transform:      float32[16] // Entity world transform
            bbox_size:      float32[3]  // Bounding box dimensions
            velocity:       float32[3]  // Entity velocity
            is_static:      bool        // Static vs dynamic flag
        }
    ]

    // === ROAD TOPOLOGY (from RoadNetworkManager) ===
    road_context: {
        closest_road_id:    uint32      // ID of nearest road segment
        road_width:         float32     // Road width in meters
        lane_centerline:    float32[20][3]  // Sampled centerline points
        road_type:          uint8       // 0=city, 1=highway, 2=rural, 3=offroad
        speed_limit_kmh:    float32     // Posted speed limit
        distance_to_junction: float32   // Distance to next junction
        junction_type:      uint8       // Intersection classification
    }

    // === ENVIRONMENT STATE ===
    environment: {
        time_of_day:        float32     // 0-24 hour decimal
        weather_state:      string[32]  // Weather preset name
        sun_direction:      float32[3]  // Normalized sun vector
        ambient_light:      float32[3]  // RGB ambient color
        fog_density:        float32     // Fog factor [0, 1]
        rain_intensity:     float32     // Rain factor [0, 1]
    }
}
```

### 1.2 Anchor Frame Designation (Self-Forcing++)

For Self-Forcing++ training, certain frames are designated as "anchor frames" that serve as reset points during long sequence generation.

```
AnchorFramePolicy {
    // Anchor frames designated every N frames
    anchor_interval:        int = 150    // Every 30 seconds at 5Hz

    // Additional anchors at scene transitions
    anchor_on_conditions: [
        "road_type_change",      // City -> Highway transition
        "junction_traversal",    // Intersection crossing
        "vehicle_stop",          // Speed < 2 km/h for > 2s
        "scene_occlusion"        // Major visibility change
    ]

    // Anchor metadata
    anchor_record: {
        anchor_id:          uint32
        frame_id:           uint64
        anchor_type:        string      // "periodic", "transition", "stop"
        context_window:     uint32      // Frames of context before anchor
        rollout_horizon:    uint32      // Max frames to generate after anchor
    }
}
```

---

## 2. Component Hierarchy

### 2.1 High-Level Component Architecture

```
SCR_WorldModelDataCapture (GameModeComponent)
    |
    +-- SCR_FrameSynchronizer
    |       Manages capture timing and buffer coordination
    |
    +-- SCR_MultiCameraRig
    |       |-- SCR_CameraCaptureUnit (x4: front, left, right, rear)
    |       |       Handles per-camera intrinsics and transform capture
    |       |
    |       +-- SCR_DepthRaycaster
    |               Grid-based depth map generation via TraceMove
    |
    +-- SCR_VehicleTelemetryCapture
    |       Interfaces with VehicleWheeledSimulation
    |       Computes velocities, accelerations
    |
    +-- SCR_SceneEnumerator
    |       QueryEntitiesBySphere for dynamic objects
    |       3D bounding box extraction
    |       Semantic classification
    |
    +-- SCR_RoadTopologyCapture
    |       Integrates with RoadNetworkManager
    |       Road graph serialization
    |
    +-- SCR_EnvironmentCapture
    |       TimeAndWeatherManagerEntity interface
    |       Lighting and atmospheric conditions
    |
    +-- SCR_AnchorFrameSelector
    |       Self-Forcing++ anchor designation logic
    |
    +-- SCR_BinarySerializer
            .enfcap container format writer
```

### 2.2 Component Specifications

#### SCR_FrameSynchronizer

**Purpose**: Maintains a stable 5Hz capture clock and coordinates buffer writes across all subsystems.

```
Attributes:
    m_fCaptureIntervalMs: float = 200.0    // 5Hz base rate
    m_iBufferDepth: int = 10               // Ring buffer size
    m_bAsyncWrite: bool = true             // Background I/O

Methods:
    void OnFrame(float dt)
        - Accumulates time delta
        - Triggers capture when interval exceeded
        - Manages frame ID assignment

    void TriggerCapture(uint64 frameId, float timestamp)
        - Signals all capture components
        - Collects responses into FrameRecord
        - Queues for serialization

    void FlushBuffers()
        - Forces pending writes to disk
        - Called on session end
```

**Synchronization Strategy**:
- All components register with the synchronizer
- Capture is triggered via callback on exact 200ms boundaries
- Components have 50ms to complete data collection
- Late data is marked with interpolation flag

#### SCR_MultiCameraRig

**Purpose**: Manages multiple camera viewpoints for surround-view capture.

```
Camera Configuration:
    Front Camera (Primary):
        - FOV: 90 degrees vertical
        - Resolution: 1920x1080 (configurable)
        - Position: [0, 1.5, 2.5] relative to vehicle
        - Orientation: Forward-facing

    Left Camera:
        - FOV: 90 degrees
        - Resolution: 1280x720
        - Position: [-1.0, 1.5, 0]
        - Orientation: -90 degrees yaw

    Right Camera:
        - FOV: 90 degrees
        - Resolution: 1280x720
        - Position: [1.0, 1.5, 0]
        - Orientation: +90 degrees yaw

    Rear Camera:
        - FOV: 120 degrees (wide for reversing)
        - Resolution: 1280x720
        - Position: [0, 1.5, -1.5]
        - Orientation: 180 degrees yaw

Intrinsic Matrix Computation:
    K = | fx  0  cx |
        | 0  fy  cy |
        | 0   0   1 |

    Where:
        fx = width / (2 * tan(fov_h / 2))
        fy = height / (2 * tan(fov_v / 2))
        cx = width / 2
        cy = height / 2
```

#### SCR_DepthRaycaster

**Purpose**: Generate dense depth maps using engine raycasting.

```
Configuration:
    m_iGridWidth: int = 192          // Horizontal ray count
    m_iGridHeight: int = 108         // Vertical ray count (16:9)
    m_fMaxDepth: float = 200.0       // Maximum trace distance (m)
    m_iLayerMask: int = 0xFFFFFF     // Trace all layers

Algorithm:
    for each (u, v) in grid:
        // Convert grid coord to normalized device coordinates
        ndc_x = (2.0 * u / width) - 1.0
        ndc_y = 1.0 - (2.0 * v / height)

        // Unproject to world ray using camera intrinsics
        ray_dir = camera_rotation * normalize(
            ndc_x / fx,
            ndc_y / fy,
            1.0
        )

        // Setup TraceParam
        trace.Start = camera_position
        trace.End = camera_position + ray_dir * max_depth
        trace.Flags = TraceFlags.WORLD | TraceFlags.ENTS
        trace.LayerMask = m_iLayerMask

        // Execute trace
        hit_fraction = world.TraceMove(trace)

        if hit_fraction < 1.0:
            depth[v][u] = hit_fraction * max_depth
            valid[v][u] = 1
            semantic[v][u] = ClassifyEntity(trace.TraceEnt)
        else:
            depth[v][u] = max_depth  // Sky/infinity
            valid[v][u] = 0
            semantic[v][u] = SEMANTIC_SKY

Performance Optimization:
    - Use AsyncTraceMove for non-blocking traces
    - Process in 32x32 tile batches
    - Cache static geometry results for 5 frames
    - Skip traces for known sky regions (top 20% of image)
```

#### SCR_SceneEnumerator

**Purpose**: Enumerate and classify all scene entities within perception range.

```
Configuration:
    m_fQueryRadius: float = 150.0    // Meters around ego vehicle
    m_iMaxEntities: int = 256        // Entity buffer limit

Entity Classification (Semantic IDs):
    0: Unknown
    1: Vehicle_Car
    2: Vehicle_Truck
    3: Vehicle_Motorcycle
    4: Vehicle_Military
    5: Pedestrian
    6: Cyclist
    7: Animal
    8: Building
    9: Vegetation
    10: TrafficSign
    11: TrafficLight
    12: Barrier
    13: Pole
    14: Ground
    15: Road
    16: Sidewalk

Methods:
    void QuerySceneEntities(vector center, float radius):
        // Use sphere query
        world.QueryEntitiesBySphere(
            center,
            radius,
            OnEntityFound,    // Callback per entity
            FilterEntity,     // Skip irrelevant
            EQueryEntitiesFlags.DYNAMIC
        )

    int ClassifyEntity(IEntity entity):
        // Check for vehicle component
        if VehicleWheeledSimulation.Cast(entity.FindComponent(...)):
            return VEHICLE_CAR

        // Check for character controller
        if CharacterControllerComponent.Cast(entity.FindComponent(...)):
            return PEDESTRIAN

        // Check entity prefab name for classification
        string name = entity.GetPrefabData().GetPrefabName()
        return ClassifyByPrefabName(name)

    BoundingBox3D ComputeBoundingBox(IEntity entity):
        vector min, max
        entity.GetBounds(min, max)
        return {
            center: (min + max) * 0.5,
            size: max - min,
            transform: entity.GetWorldTransform()
        }
```

#### SCR_RoadTopologyCapture

**Purpose**: Capture road network context around the ego vehicle.

```
Configuration:
    m_fSearchRadius: float = 200.0   // Road query radius
    m_iMaxRoads: int = 50            // Max roads to process
    m_iCenterlinePoints: int = 20    // Points per road segment

Integration with RoadNetworkManager:
    void CaptureRoadContext(vector egoPosition):
        // Get nearby roads
        array<BaseRoad> nearbyRoads
        vector min = egoPosition - Vector(radius, radius, radius)
        vector max = egoPosition + Vector(radius, radius, radius)
        m_RoadNetworkManager.GetRoadsInAABB(min, max, nearbyRoads)

        // Find closest road
        BaseRoad closestRoad
        float closestDist
        m_RoadNetworkManager.GetClosestRoad(
            egoPosition,
            closestRoad,
            closestDist
        )

        // Extract road properties
        if closestRoad:
            array<vector> points
            closestRoad.GetPoints(points)
            float width = closestRoad.GetWidth()

            // Sample centerline at regular intervals
            centerline = ResamplePoints(points, m_iCenterlinePoints)

            // Classify road type from context
            roadType = ClassifyRoadFromWidth(width)

Road Classification Heuristics:
    width > 10m          -> Highway (2+ lanes each direction)
    width 6-10m          -> Urban road (2 lanes)
    width 3-6m           -> Rural road (single lane)
    width < 3m           -> Path/offroad

Junction Detection:
    - Parse road endpoint proximity (< 3m tolerance)
    - Count connected roads at junction point
    - Classify: T-junction (3), Crossroads (4), Roundabout (5+)
```

#### SCR_AnchorFrameSelector

**Purpose**: Designate anchor frames for Self-Forcing++ training.

```
State Machine:
    States:
        NORMAL          // Regular capture mode
        TRANSITION      // Scene change detected
        ANCHOR_PENDING  // Waiting to write anchor

Anchor Triggers:
    1. Periodic Anchor:
        - Every 150 frames (30 seconds at 5Hz)
        - Ensures maximum rollout segment length

    2. Road Type Change:
        - Detect road_type transition
        - Mark anchor 5 frames before change
        - Provides context for transition

    3. Junction Traversal:
        - Detect entry into junction zone
        - Mark anchor at junction approach
        - Captures decision point

    4. Vehicle State Change:
        - Speed drops below 2 km/h for > 2 seconds (stop)
        - Speed increases from stop (start)
        - Gear change R <-> D

    5. Significant Occlusion:
        - Track visible entity count
        - Anchor when > 30% change in visible entities
        - Handles building occlusion scenarios

Anchor Metadata:
    anchor_record = {
        anchor_id: sequential_counter,
        frame_id: current_frame,
        anchor_type: trigger_type,
        context_window: 50,      // 10 seconds prior
        rollout_horizon: 150,    // 30 seconds forward
        scene_hash: compute_scene_hash()  // For retrieval augmentation
    }
```

---

## 3. Capture Rates and Synchronization

### 3.1 Timing Hierarchy

```
+----------------------------------------------------------+
|                    CAPTURE TIMING                        |
+----------------------------------------------------------+
| Layer          | Rate    | Interval | Purpose            |
+----------------------------------------------------------+
| Base Clock     | 5 Hz    | 200ms    | Frame capture      |
| Vehicle State  | 5 Hz    | 200ms    | Telemetry sync     |
| Depth Maps     | 2.5 Hz  | 400ms    | Performance opt.   |
| Scene Query    | 5 Hz    | 200ms    | Entity tracking    |
| Road Topology  | 1 Hz    | 1000ms   | Low volatility     |
| Environment    | 0.2 Hz  | 5000ms   | Rare changes       |
| Anchor Check   | 5 Hz    | 200ms    | Every frame        |
+----------------------------------------------------------+
```

### 3.2 Synchronization Protocol

```
Frame Capture Sequence (per 200ms tick):

T+0ms:    FrameSynchronizer.TriggerCapture()
          |
          +-> Signal all components with frame_id

T+5ms:    SCR_VehicleTelemetryCapture completes
          - Reads VehicleWheeledSimulation state
          - Computes derived values (acceleration, etc.)

T+10ms:   SCR_MultiCameraRig completes
          - Samples camera transforms
          - Computes intrinsic matrices

T+30ms:   SCR_DepthRaycaster completes (every other frame)
          - 192x108 raycasts complete
          - Depth buffer populated

T+40ms:   SCR_SceneEnumerator completes
          - Entity query finished
          - Bounding boxes computed

T+45ms:   SCR_AnchorFrameSelector evaluates
          - Checks anchor conditions
          - Updates anchor state

T+50ms:   FrameRecord assembly
          - All component data aggregated
          - Timestamp validation

T+55ms:   Async serialization queued
          - Binary packing begins
          - I/O thread handles write

Synchronization Guarantees:
    - All data within frame has same timestamp (within 50ms)
    - Frame IDs monotonically increase
    - No frame drops (buffer underrun triggers session pause)
    - Interpolation flag set if any component late
```

### 3.3 Buffer Management

```
Ring Buffer Architecture:

+-------+-------+-------+-------+-------+-------+-------+-------+
| F[n-7]| F[n-6]| F[n-5]| F[n-4]| F[n-3]| F[n-2]| F[n-1]| F[n]  |
+-------+-------+-------+-------+-------+-------+-------+-------+
        ^                               ^               ^
        |                               |               |
    Serialize                      Aggregate        Capture
    (async write)                  (assembly)       (current)

Buffer Sizing:
    - 8-frame ring buffer
    - Total memory: ~50MB per buffer slot
    - Allows 1.6 seconds of capture latency tolerance

Overflow Policy:
    - On buffer full: pause capture, flush to disk
    - Resume after 50% buffer free
    - Log warning with frame count gap
```

---

## 4. Depth Capture via Raycasting

### 4.1 Raycast Grid Design

```
Depth Map Resolution Strategy:

Full Resolution (Optional, High Perf Cost):
    1920 x 1080 = 2,073,600 rays
    At 200us per ray = 414 seconds (NOT FEASIBLE)

Practical Resolution:
    192 x 108 = 20,736 rays
    At 200us per ray = 4.15 seconds
    With async + batching = ~50ms achievable

Subsampled Grid with Upscaling:
    96 x 54 base grid = 5,184 rays
    Bilinear upscale to 192 x 108
    Edge-aware refinement for object boundaries
    Final latency: ~15ms
```

### 4.2 Raycast Implementation

```
Depth Capture Algorithm:

class SCR_DepthRaycaster {
    // Pre-computed ray directions (computed once at init)
    ref array<vector> m_aRayDirections;

    // Output buffers
    ref array<float> m_aDepthBuffer;
    ref array<int> m_aSemanticBuffer;

    void Initialize(CameraParams camera):
        // Pre-compute normalized ray directions
        for v in range(height):
            for u in range(width):
                // NDC coordinates
                ndc_x = (2.0 * u + 1.0) / width - 1.0
                ndc_y = 1.0 - (2.0 * v + 1.0) / height

                // Unproject using camera intrinsics
                dir = vector(
                    ndc_x / camera.fx,
                    ndc_y / camera.fy,
                    1.0
                )
                dir.Normalize()
                m_aRayDirections.Insert(dir)

    void CaptureDepth(vector camPos, vector camMat[4]):
        TraceParam trace = new TraceParam()
        trace.Flags = TraceFlags.WORLD | TraceFlags.ENTS
        trace.LayerMask = EPhysicsLayerPresets.Projectile

        for i in range(m_aRayDirections.Count()):
            // Transform ray to world space
            vector worldDir = camMat[0] * m_aRayDirections[i][0]
                            + camMat[1] * m_aRayDirections[i][1]
                            + camMat[2] * m_aRayDirections[i][2]

            trace.Start = camPos
            trace.End = camPos + worldDir * m_fMaxDepth

            float hit = world.TraceMove(trace)

            if hit < 1.0:
                m_aDepthBuffer[i] = hit * m_fMaxDepth
                m_aSemanticBuffer[i] = ClassifyHit(trace.TraceEnt)
            else:
                m_aDepthBuffer[i] = m_fMaxDepth
                m_aSemanticBuffer[i] = SEMANTIC_SKY
}

Optimization Techniques:
    1. Batch Processing: Process rays in 32x32 tiles
    2. Async Traces: Use AsyncTraceMove for non-blocking
    3. Temporal Caching: Skip static geometry re-traces
    4. Frustum Culling: Skip rays pointing at known sky
    5. LOD: Reduce ray density for distant regions
```

### 4.3 Depth Data Format

```
Depth Map Encoding:

Primary Format: float16 (IEEE 754 half-precision)
    Range: 0.0 to 200.0 meters
    Precision: ~0.1m at 100m distance
    Size: 192 x 108 x 2 bytes = 41.5 KB per camera

Validity Mask: uint8 per pixel
    0 = Invalid (sky, max range)
    1 = Valid measurement
    Size: 192 x 108 x 1 byte = 20.7 KB per camera

Semantic Labels: uint8 per pixel
    Encodes entity class ID (0-255)
    Size: 192 x 108 x 1 byte = 20.7 KB per camera

Total Depth Data Per Camera: ~83 KB
Total for 4 Cameras: ~332 KB per frame
```

---

## 5. Multi-Camera Support

### 5.1 Camera Rig Configuration

```
+------------------+
|       REAR       |   Camera 3: Wide rear view
|    [120 FOV]     |
+--------+---------+
         |
+--------+---------+--------+
|  LEFT  |  FRONT  | RIGHT  |
|[90 FOV]|[90 FOV] |[90 FOV]|
+--------+---------+--------+
    ^         ^         ^
    |         |         |
 Cam 1     Cam 0     Cam 2

Camera Mount Points (relative to vehicle center):
    Camera 0 (Front):  [ 0.0,  1.5,  2.5] heading 0
    Camera 1 (Left):   [-1.0,  1.5,  0.0] heading -90
    Camera 2 (Right):  [ 1.0,  1.5,  0.0] heading +90
    Camera 3 (Rear):   [ 0.0,  1.5, -1.5] heading 180
```

### 5.2 Camera Component Implementation

```
class SCR_CameraCaptureUnit {
    // Configuration
    int m_iCameraId;
    float m_fVerticalFov;
    float m_fAspectRatio;
    vector m_vMountOffset;
    vector m_vMountRotation;  // Euler angles

    // Computed values
    ref float m_aIntrinsicMatrix[9];
    ref float m_aExtrinsicMatrix[16];

    void ComputeIntrinsicMatrix(int width, int height):
        float fov_h = 2.0 * Math.Atan(
            Math.Tan(m_fVerticalFov * 0.5 * DEG2RAD) * m_fAspectRatio
        )

        float fx = width / (2.0 * Math.Tan(fov_h * 0.5))
        float fy = height / (2.0 * Math.Tan(m_fVerticalFov * 0.5 * DEG2RAD))
        float cx = width * 0.5
        float cy = height * 0.5

        // Row-major 3x3 intrinsic matrix
        m_aIntrinsicMatrix[0] = fx;  m_aIntrinsicMatrix[1] = 0;  m_aIntrinsicMatrix[2] = cx;
        m_aIntrinsicMatrix[3] = 0;   m_aIntrinsicMatrix[4] = fy; m_aIntrinsicMatrix[5] = cy;
        m_aIntrinsicMatrix[6] = 0;   m_aIntrinsicMatrix[7] = 0;  m_aIntrinsicMatrix[8] = 1;

    void ComputeExtrinsicMatrix(vector vehicleTransform[4]):
        // Compute camera world transform from vehicle transform
        // 1. Apply mount offset in vehicle space
        // 2. Apply mount rotation
        // 3. Combine with vehicle world transform

        vector camLocalMat[4];
        Math3D.MatrixIdentity3(camLocalMat);
        Math3D.MatrixRotationYawPitchRoll(
            camLocalMat,
            m_vMountRotation[0], m_vMountRotation[1], m_vMountRotation[2]
        );
        camLocalMat[3] = m_vMountOffset;

        vector camWorldMat[4];
        Math3D.MatrixMultiply4(vehicleTransform, camLocalMat, camWorldMat);

        // Flatten to 16-float array (row-major)
        for i in range(4):
            for j in range(4):
                m_aExtrinsicMatrix[i*4 + j] = camWorldMat[i][j];
}
```

### 5.3 Multi-Camera Synchronization

```
Capture Order (per frame):

1. Sample vehicle transform (single source of truth)
2. Compute all camera extrinsics from vehicle transform
3. Capture depth for each camera (parallelizable)
4. Package into multi-camera frame record

Calibration Data (stored once per session):
    - Mount offsets and rotations
    - Lens distortion coefficients (if applicable)
    - Stereo baseline for camera pairs

Frame Data Layout:
    FrameRecord {
        ...
        cameras[4]: {
            camera_id,
            intrinsic[9],
            extrinsic[16],
            depth_map[H][W],
            semantic_map[H][W]
        }
    }
```

---

## 6. Road Topology Integration

### 6.1 Integration with Existing Extractors

```
Leveraging SCR_EfficientRoadNet / SCR_LocalRoadExtractor:

The existing road visualization components provide:
    - RoadNetworkManager access
    - Road point extraction
    - Junction detection
    - Path prediction

Road Topology Capture Strategy:
    1. Reuse existing RoadNetworkManager initialization
    2. Extract road data on 1Hz cadence (low volatility)
    3. Cache road graph for session duration
    4. Update local context each frame
```

### 6.2 Road Graph Serialization

```
Binary Road Graph Format (.roadgraph):

Header:
    magic:          uint32 = 0x524F4144 ("ROAD")
    version:        uint16 = 1
    num_roads:      uint32
    num_junctions:  uint32
    bounds_min:     float32[3]
    bounds_max:     float32[3]

Road Segment Record:
    road_id:        uint32
    width:          float32
    road_type:      uint8
    num_points:     uint16
    points:         float32[num_points][3]
    start_junction: uint32 (0xFFFFFFFF if none)
    end_junction:   uint32 (0xFFFFFFFF if none)

Junction Record:
    junction_id:    uint32
    position:       float32[3]
    junction_type:  uint8
    num_connected:  uint8
    connected_roads: uint32[num_connected]
```

### 6.3 Per-Frame Road Context

```
Road Context Extraction (per frame):

class SCR_RoadContextCapture {
    void CaptureContext(vector egoPos, vector egoFwd):
        // 1. Find closest road
        BaseRoad closestRoad;
        float distance;
        m_RoadNetworkManager.GetClosestRoad(egoPos, closestRoad, distance);

        // 2. Sample centerline in direction of travel
        array<vector> roadPoints;
        closestRoad.GetPoints(roadPoints);

        // 3. Find closest point and direction
        int closestIdx = FindClosestPoint(roadPoints, egoPos);
        bool forward = DetermineDirection(roadPoints, closestIdx, egoFwd);

        // 4. Extract future centerline (20 points ahead)
        array<vector> futureCenterline = SampleFuturePath(
            roadPoints, closestIdx, forward, 20
        );

        // 5. Compute road features
        float width = closestRoad.GetWidth();
        int roadType = ClassifyRoad(width, distance);
        float distToJunction = ComputeJunctionDistance(
            roadPoints, closestIdx, forward
        );

        return RoadContext {
            closest_road_id: closestRoad.GetID(),
            road_width: width,
            road_type: roadType,
            lane_centerline: futureCenterline,
            distance_to_junction: distToJunction
        };
}
```

---

## 7. Binary File Format Specification

### 7.1 Container Format (.enfcap)

```
ENFCAP File Structure:

+------------------------+
|     FILE HEADER        |  32 bytes
+------------------------+
|     CHUNK INDEX        |  Variable (8 bytes per chunk)
+------------------------+
|     CHUNK DATA         |  Variable
|     (Frame chunks)     |
+------------------------+
|     FOOTER             |  16 bytes
+------------------------+

File Header (32 bytes):
    magic:              uint32 = 0x454E4643 ("ENFC")
    version:            uint16 = 1
    flags:              uint16 (compression flags)
    session_id:         uint32
    start_timestamp:    float64
    num_chunks:         uint32
    chunk_index_offset: uint64
    reserved:           uint32

Chunk Index Entry (8 bytes per chunk):
    chunk_offset:       uint32 (relative to chunk data start)
    chunk_size:         uint32 (compressed size)

Chunk Data (per frame):
    chunk_header:       16 bytes
        frame_id:       uint64
        timestamp:      float64

    telemetry_block:    variable
        ego_transform:  64 bytes (16 x float32)
        vehicle_state:  48 bytes (speed, steering, etc.)
        control_input:  16 bytes (4 x float32)

    camera_block[4]:    variable per camera
        intrinsic:      36 bytes (9 x float32)
        extrinsic:      64 bytes (16 x float32)
        depth_data:     compressed (LZ4)
        semantic_data:  compressed (LZ4)

    entity_block:       variable
        num_entities:   uint16
        entities:       entity_record[]

    road_block:         variable
        context_data:   road_context_record

    environment_block:  64 bytes
        time_weather:   environment_record

    anchor_block:       16 bytes (if anchor frame)
        anchor_record:  anchor_metadata

Footer (16 bytes):
    checksum:           uint64 (CRC64 of all chunks)
    total_frames:       uint32
    end_magic:          uint32 = 0x454E4445 ("ENDE")
```

### 7.2 Compression Strategy

```
Compression by Data Type:

Data Type           | Compression | Rationale
--------------------|-------------|---------------------------
Telemetry floats    | None        | Small size, need precision
Camera intrinsics   | None        | Fixed size, rare change
Camera extrinsics   | None        | Small, every frame
Depth maps          | LZ4         | High compressibility
Semantic maps       | LZ4         | Very high compressibility
Entity data         | LZ4         | Variable size
Road context        | None        | Small, fixed structure
Anchor metadata     | None        | Tiny, critical data

Compression Ratios (typical):
    Depth maps:     3:1 (smooth gradients compress well)
    Semantic maps:  10:1 (large uniform regions)
    Entity blocks:  2:1 (structured data)

Overall file size per frame:
    Uncompressed: ~400 KB
    Compressed:   ~150 KB

Session storage (1 hour at 5Hz):
    Frames: 18,000
    Total: ~2.7 GB compressed
```

### 7.3 Serialization Implementation

```
class SCR_BinarySerializer {
    // File handle
    FileHandle m_File;

    // Chunk buffer
    ref array<uint8> m_aChunkBuffer;

    // Index tracking
    ref array<uint64> m_aChunkOffsets;
    ref array<uint32> m_aChunkSizes;

    bool BeginSession(string sessionPath):
        m_File = FileIO.OpenFile(sessionPath + "/frames.enfcap", FileMode.WRITE);
        if (!m_File) return false;

        // Write placeholder header
        WriteHeader(0, 0);  // Will update at end

        // Reserve space for chunk index (estimate max frames)
        m_ChunkIndexOffset = 32;
        return true;

    void WriteFrame(FrameRecord frame):
        // Serialize to buffer
        int offset = 0;

        // Frame header
        WriteUInt64(m_aChunkBuffer, offset, frame.frame_id);
        WriteFloat64(m_aChunkBuffer, offset, frame.timestamp_ms);

        // Telemetry block
        WriteFloatArray(m_aChunkBuffer, offset, frame.ego_transform, 16);
        WriteVehicleState(m_aChunkBuffer, offset, frame);

        // Camera blocks
        for each camera in frame.cameras:
            WriteFloatArray(m_aChunkBuffer, offset, camera.intrinsic, 9);
            WriteFloatArray(m_aChunkBuffer, offset, camera.extrinsic, 16);

            // Compress depth
            array<uint8> compressedDepth = LZ4.Compress(camera.depth_map);
            WriteUInt32(m_aChunkBuffer, offset, compressedDepth.Count());
            WriteByteArray(m_aChunkBuffer, offset, compressedDepth);

            // Compress semantic
            array<uint8> compressedSemantic = LZ4.Compress(camera.semantic_map);
            WriteUInt32(m_aChunkBuffer, offset, compressedSemantic.Count());
            WriteByteArray(m_aChunkBuffer, offset, compressedSemantic);

        // Entity block
        WriteEntityBlock(m_aChunkBuffer, offset, frame.entities);

        // Road block
        WriteRoadContext(m_aChunkBuffer, offset, frame.road_context);

        // Environment block
        WriteEnvironment(m_aChunkBuffer, offset, frame.environment);

        // Anchor block (if applicable)
        if frame.is_anchor_frame:
            WriteAnchorRecord(m_aChunkBuffer, offset, frame.anchor);

        // Track chunk for index
        m_aChunkOffsets.Insert(m_File.GetPos());
        m_aChunkSizes.Insert(offset);

        // Write chunk
        m_File.Write(m_aChunkBuffer, offset);

    void EndSession():
        // Write chunk index
        int indexOffset = m_File.GetPos();
        for i in range(m_aChunkOffsets.Count()):
            m_File.Write(m_aChunkOffsets[i], 8);
            m_File.Write(m_aChunkSizes[i], 4);

        // Write footer
        WriteFooter(m_aChunkOffsets.Count());

        // Update header with final values
        m_File.Seek(0);
        WriteHeader(m_aChunkOffsets.Count(), indexOffset);

        m_File.Close();
}
```

---

## 8. Performance Considerations

### 8.1 Performance Budget

```
Frame Budget at 5Hz (200ms per frame):

Component               | Budget  | Actual (est.) | Notes
------------------------|---------|---------------|------------------
Vehicle telemetry       | 5ms     | <1ms          | Direct API calls
Camera intrinsics       | 2ms     | <1ms          | Computed once
Depth raycasting        | 80ms    | 50ms          | 20K rays, batched
Scene enumeration       | 20ms    | 15ms          | Sphere query
Road context            | 10ms    | 5ms           | Cached graph
Environment capture     | 2ms     | <1ms          | API calls
Anchor evaluation       | 2ms     | <1ms          | State machine
Frame assembly          | 10ms    | 5ms           | Memory copy
Serialization (async)   | N/A     | 20ms          | Background thread
------------------------|---------|---------------|------------------
TOTAL                   | 131ms   | ~98ms         | 50% headroom
```

### 8.2 Memory Budget

```
Runtime Memory Usage:

Component                   | Size      | Notes
----------------------------|-----------|---------------------------
Frame ring buffer (8 slots) | 400 MB    | 50MB per frame record
Chunk write buffer          | 50 MB     | Double-buffered
Ray direction cache         | 320 KB    | 20K vectors
Road graph cache            | 10 MB     | Full local graph
Entity tracking buffer      | 2 MB      | 256 entities max
Depth buffer (4 cameras)    | 660 KB    | 192x108x4 bytes x4
Semantic buffer (4 cameras) | 165 KB    | 192x108x1 byte x4
----------------------------|-----------|---------------------------
TOTAL                       | ~465 MB   | Acceptable for modding
```

### 8.3 I/O Optimization

```
Asynchronous I/O Strategy:

Write Thread:
    - Dedicated background thread for file I/O
    - Pulls from serialization queue
    - Batch writes of 10 frames minimum
    - Flush on queue empty or 1 second timeout

Buffer Management:
    - Ping-pong buffers for serialization
    - Main thread writes to active buffer
    - I/O thread reads from standby buffer
    - Swap on frame boundary

Disk Requirements:
    - Sustained write: ~750 KB/s (150KB x 5 frames)
    - Burst write: 1.5 MB/s (on flush)
    - Any SSD easily handles this load
    - HDD viable with larger buffers
```

### 8.4 Graceful Degradation

```
Performance Scaling Options:

Level 0 (Full Quality):
    - 4 cameras, 192x108 depth
    - 5Hz capture rate
    - All data streams enabled

Level 1 (Reduced Depth):
    - 4 cameras, 96x54 depth (upscaled)
    - 5Hz capture rate
    - All data streams enabled

Level 2 (Reduced Cameras):
    - 2 cameras (front + rear)
    - 96x54 depth
    - 5Hz capture rate

Level 3 (Reduced Rate):
    - 2 cameras
    - 96x54 depth
    - 2.5Hz capture rate

Level 4 (Telemetry Only):
    - No depth capture
    - Telemetry + entities only
    - 5Hz capture rate

Auto-Scaling Trigger:
    - Monitor frame assembly time
    - If > 150ms, drop to lower level
    - If < 80ms for 60 frames, try higher level
```

---

## 9. Integration Points

### 9.1 Training Pipeline Integration

```
Data Loading for PyTorch:

class EnfusionDataset(torch.utils.data.Dataset):
    def __init__(self, enfcap_path):
        self.reader = EnfcapReader(enfcap_path)
        self.index = self.reader.load_chunk_index()

    def __getitem__(self, idx):
        frame = self.reader.read_frame(self.index[idx])

        # Decompress depth maps
        depth_maps = [
            lz4.decompress(cam.depth_data).reshape(108, 192)
            for cam in frame.cameras
        ]

        # Build training sample
        return {
            'ego_transform': frame.ego_transform,
            'vehicle_state': frame.vehicle_state,
            'depth': np.stack(depth_maps),
            'road_context': frame.road_context,
            'is_anchor': frame.is_anchor_frame
        }

Sequence Loading for Self-Forcing:
    def load_sequence(self, start_idx, length):
        frames = [self[start_idx + i] for i in range(length)]

        # Find anchor frames in sequence
        anchors = [i for i, f in enumerate(frames) if f['is_anchor']]

        return {
            'frames': frames,
            'anchor_indices': anchors
        }
```

### 9.2 Manifest File Format

```json
{
    "version": "1.0",
    "session_id": "20260323_143052",
    "capture_config": {
        "base_rate_hz": 5,
        "depth_resolution": [192, 108],
        "num_cameras": 4,
        "max_depth_m": 200.0
    },
    "environment": {
        "map_name": "GM_Arland",
        "initial_weather": "Clear",
        "initial_time": 12.0
    },
    "statistics": {
        "total_frames": 18247,
        "duration_seconds": 3649.4,
        "total_distance_m": 45632.7,
        "anchor_frames": 122,
        "road_types_covered": ["city", "highway", "rural"],
        "weather_changes": ["Clear", "Cloudy", "Rain"]
    },
    "files": {
        "frames": "frames.enfcap",
        "road_graph": "road_topology.bin",
        "manifest": "manifest.json"
    },
    "schema_version": "GAIA2_COMMA_COMPAT_V1"
}
```

---

## 10. Future Extensions

### 10.1 Semantic Segmentation

```
Future: Full semantic map capture using material queries

Per-pixel material sampling:
    - Query ContactSurface for hit materials
    - Map materials to semantic classes
    - Higher resolution (384x216) with tiling

Integration with DINOv2/v3:
    - Export RGB frames alongside depth
    - Use DINOv2 for feature extraction
    - Dense feature maps for V-JEPA style training
```

### 10.2 LiDAR Simulation

```
Future: Simulated LiDAR point clouds

LiDAR Pattern:
    - 64-beam Velodyne HDL-64E simulation
    - 360-degree sweep at 10Hz
    - 2.6M points per second

Implementation:
    - Spherical raycast pattern
    - Return (x, y, z, intensity)
    - Export as .las or custom binary
```

### 10.3 Multi-Agent Capture

```
Future: Capture from multiple ego vehicles

Coordination:
    - Shared FrameSynchronizer across vehicles
    - Per-vehicle SCR_WorldModelDataCapture instance
    - Unified session with vehicle_id field

Use Case:
    - Multi-agent driving scenarios
    - V2V communication simulation
    - Fleet training data
```

---

## Appendix A: API Reference Summary

### Enfusion Engine APIs Used

| API | Purpose |
|-----|---------|
| `VehicleWheeledSimulation` | Vehicle dynamics state |
| `CameraBase.GetWorldCameraTransform()` | Camera pose |
| `CameraBase.GetVerticalFOV()` | Camera intrinsics |
| `TraceParam` / `world.TraceMove()` | Raycasting for depth |
| `world.QueryEntitiesBySphere()` | Entity enumeration |
| `RoadNetworkManager.GetRoadsInAABB()` | Road topology |
| `RoadNetworkManager.GetClosestRoad()` | Road context |
| `BaseRoad.GetPoints()` / `.GetWidth()` | Road geometry |
| `TimeAndWeatherManagerEntity` | Environment state |
| `FileIO.OpenFile()` / `FileHandle` | Binary I/O |
| `PerceptionComponent.GetTargetsList()` | AI perception targets |

### Key Enfusion Limitations

1. **No Direct Screenshot API**: Must use external tools for RGB capture
2. **No GPU Compute**: All raycasting is CPU-bound
3. **Limited FileIO**: Only `$profile:`, `$logs:`, `$saves:` writable
4. **No Threading API**: Async operations via engine callbacks only
5. **No Compression API**: LZ4 must be implemented in script or omitted

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| Anchor Frame | Self-Forcing++ reset point for long sequence generation |
| ENFCAP | Enfusion Capture format - binary container for frame data |
| Extrinsic Matrix | 4x4 camera-to-world transformation |
| Intrinsic Matrix | 3x3 camera projection parameters |
| KV Cache | Key-Value cache for transformer attention layers |
| Self-Forcing++ | Training method using self-generated outputs |
| Telemetry | Vehicle dynamics and control signal data |
| V-JEPA | Video Joint-Embedding Predictive Architecture |

---

*Document Version: 1.0*
*Last Updated: 2026-03-23*
*Author: DriveDiT Architecture Team*
