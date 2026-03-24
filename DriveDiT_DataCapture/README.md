# DriveDiT Data Capture Mod

**ML Training Data Capture System for Autonomous Driving World Models**

This Arma Reforger mod captures synchronized vehicle telemetry, depth data, and scene information for training diffusion transformer models like DriveDiT.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output Data Format](#output-data-format)
- [Components](#components)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Features

- **Vehicle Telemetry Capture**: Position, orientation, speed, steering, throttle, brake, gear, RPM
- **Depth Map Generation**: Grid-based raycasting depth capture (configurable resolution)
- **Scene Enumeration**: Object tracking and classification for ground truth labels
- **Multiple Capture Profiles**: Minimal, Research, and Production configurations
- **AI Driving Simulation**: Automated data collection with AI-driven vehicles
- **Binary Serialization**: Efficient data formats for ML training pipelines
- **Debug UI**: Real-time visualization of capture status and telemetry

## Requirements

### Game Requirements
- Arma Reforger (Steam version recommended)
- Arma Reforger Tools (for development/building)

### System Requirements
- Windows 10/11
- 16GB RAM (recommended)
- SSD storage for data output
- Python 3.8+ (for build scripts)

## Installation

### From Workshop (End Users)

1. Subscribe to the mod on Arma Reforger Workshop
2. Enable the mod in the Arma Reforger launcher
3. Launch the game with the mod enabled

### From Source (Developers)

1. Clone or download this repository

2. Copy the `DriveDiT_DataCapture` folder to your Workbench addons directory:
   ```
   Documents/My Games/ArmaReforgerWorkbench/addons/DriveDiT_DataCapture/
   ```

3. Open Arma Reforger Workbench

4. Add the project:
   - Click "Add Existing Project"
   - Navigate to `DriveDiT_DataCapture.gproj`
   - Select and open

5. Build the mod:
   - Right-click the project
   - Select "Build"
   - Choose target platform (PC)

### Validation

Run the validation script to check the mod structure:

```bash
cd DriveDiT_DataCapture
python tools/validate.py
```

## Quick Start

### Basic Usage

1. Load a scenario with the DataCaptureGameMode prefab
2. The capture system auto-starts with default settings
3. AI vehicles will begin driving and collecting data
4. Data is saved to `$profile:DriveDiT_Captures/`

### Using in Custom Scenarios

Add the following components to your GameMode entity:

1. `SCR_CaptureOrchestrator` - Main coordinator
2. `SCR_MLDataCollector` - Telemetry capture
3. `SCR_AIDrivingSimulator` - AI vehicle management
4. `SCR_DepthRaycaster` - Depth capture (optional)
5. `SCR_SceneEnumerator` - Object tracking (optional)

### Minimal Setup

```
GenericEntity {
 components {
  SCR_BaseGameMode "{...}" {
  }
  SCR_CaptureOrchestrator "{...}" {
   m_eCaptureProfile "MINIMAL"
   m_bAutoStartCapture 1
  }
  SCR_MLDataCollector "{...}" {
   m_bEnableDataCapture 1
   m_iCaptureIntervalMs 200
  }
 }
}
```

## Configuration

### Capture Profiles

| Profile | Use Case | Data Streams | Performance Impact |
|---------|----------|--------------|-------------------|
| **Minimal** | Testing, long recordings | Telemetry only | Very Low (<5% CPU) |
| **Research** | Model development | Telemetry + Depth + Scene | Moderate (10-20% CPU) |
| **Production** | Final dataset creation | All streams | High (30-50% CPU) |

### Profile Configuration Files

- `Configs/CaptureProfiles/minimal.conf` - Telemetry only
- `Configs/CaptureProfiles/research.conf` - Balanced capture
- `Configs/CaptureProfiles/production.conf` - Full capture

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `captureIntervalMs` | 100 | Capture rate (100ms = 10Hz) |
| `maxVehicles` | 5 | Maximum tracked vehicles |
| `depthResolution` | 64x48 | Depth map resolution |
| `maxDepth` | 500m | Maximum depth distance |

## Output Data Format

### Directory Structure

```
$profile:DriveDiT_Captures/
└── session_res_1234567890/
    ├── manifest.json       # Session metadata
    ├── telemetry.csv       # Vehicle telemetry
    ├── session_info.txt    # Configuration details
    ├── summary.json        # Session statistics
    ├── depth/              # Depth maps (if enabled)
    │   ├── frame_000001.csv
    │   └── ...
    └── scene/              # Scene data (if enabled)
        ├── frame_000001.csv
        └── ...
```

### Telemetry CSV Format

```csv
frame_id,timestamp_ms,pos_x,pos_y,pos_z,fwd_x,fwd_y,fwd_z,up_x,up_y,up_z,right_x,right_y,right_z,speed_kmh,steering,throttle,brake,clutch,gear,engine_rpm,engine_on,handbrake,acceleration_kmh_s,distance_total_m,waypoint_type,speed_limit_kmh,waypoint_idx,vehicle_id
```

### Depth Map Format

- Resolution: Configurable (64x48, 128x96, etc.)
- Format: CSV with float values
- Range: 0.0 (near) to max_depth (far), -1.0 for sky/no hit
- Normalized option: 0-1 range

## Components

### SCR_CaptureOrchestrator

Central coordinator that manages all capture components.

**Responsibilities:**
- Initialize and coordinate capture components
- Manage sessions (start, pause, resume, stop)
- Synchronize timestamps across data streams
- Monitor capture health and performance

### SCR_MLDataCollector

Captures synchronized vehicle telemetry data.

**Output:**
- Position (world coordinates)
- Orientation (forward, up, right vectors)
- Control inputs (steering, throttle, brake)
- Vehicle state (gear, RPM, engine status)

### SCR_AIDrivingSimulator

Manages AI vehicles for automated data collection.

**Features:**
- Spawns configurable number of AI groups
- Manages waypoint navigation
- Tracks vehicle statistics
- Handles stuck vehicle recovery

### SCR_DepthRaycaster

Generates depth maps via raycasting.

**Features:**
- Configurable grid resolution
- Pre-computed ray directions
- Temporal smoothing
- Checkerboard optimization

### SCR_SceneEnumerator

Enumerates and tracks scene objects.

**Features:**
- Object classification (vehicles, characters, buildings)
- Bounding box extraction
- Visibility tracking
- Distance filtering

### SCR_BinarySerializer

Efficient binary data serialization.

**Formats:**
- Raw binary (int/float arrays)
- NPY-compatible (NumPy format)
- Structured (with headers)

## Development

### Building from Source

```bash
# Validate mod structure
python tools/validate.py

# Build for PC platform
python tools/build.py --platform PC --config release

# Build with validation and packaging
python tools/build.py --clean --validate --package
```

### Running Tests

```bash
# Validate mod structure
python tools/validate.py --strict --report validation_report.json
```

### Project Structure

```
DriveDiT_DataCapture/
├── DriveDiT_DataCapture.gproj    # Project file
├── Scripts/
│   └── Game/
│       └── DataCapture/          # Component scripts
│           ├── SCR_CaptureOrchestrator.c
│           ├── SCR_MLDataCollector.c
│           ├── SCR_AIDrivingSimulator.c
│           ├── SCR_DepthRaycaster.c
│           ├── SCR_SceneEnumerator.c
│           ├── SCR_BinarySerializer.c
│           └── SCR_DrivingSimDebugUI.c
├── Prefabs/
│   └── DataCapture/
│       └── DataCaptureGameMode.et
├── Configs/
│   └── CaptureProfiles/
│       ├── minimal.conf
│       ├── research.conf
│       └── production.conf
├── tools/                         # Build/validation scripts
│   ├── validate.py
│   └── build.py
└── README.md
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run validation: `python tools/validate.py`
5. Submit a pull request

## Troubleshooting

### Common Issues

**"No roads found" warning**
- Ensure the map has road network data
- Check that SCR_AIWorld is present in the world

**"Failed to initialize data collector"**
- Verify write permissions to profile directory
- Check available disk space

**Low capture frame rate**
- Reduce depth resolution
- Switch to minimal capture profile
- Reduce number of tracked vehicles

**AI vehicles stuck**
- Verify waypoint entities exist (AI_Drive_Target_1, etc.)
- Check spawn point accessibility
- Increase waypoint radius

### Debug Mode

Enable debug UI in the DiagMenu:
1. Press F8 to open DiagMenu
2. Navigate to: AIScript > DrivingSim
3. Enable "Show Telemetry Panel"

### Log Files

Check logs for detailed information:
- Console output during gameplay
- `$profile:console.log` for script errors

## License

This mod is released under the MIT License. See LICENSE file for details.

## Acknowledgments

- DriveDiT Project Team
- Bohemia Interactive for Arma Reforger
- comma.ai for openpilot/world model inspiration
