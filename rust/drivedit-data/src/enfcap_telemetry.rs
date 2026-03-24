//! ENFCAP Telemetry Format Parser
//!
//! High-performance Rust implementation of the ENFCAP telemetry binary format
//! from SCR_BinarySerializer.c. This format stores vehicle telemetry, scene entities,
//! and road topology data for ML training.
//!
//! # File Format Specification (ENFCAP Telemetry v1)
//!
//! ```text
//! +-------------------+
//! | HEADER (64 bytes) |
//! +-------------------+
//! |   FRAME 0 DATA    |
//! +-------------------+
//! |   FRAME 1 DATA    |
//! +-------------------+
//! |       ...         |
//! +-------------------+
//! |   FRAME N DATA    |
//! +-------------------+
//! |   INDEX TABLE     |
//! | (N * 8 bytes)     |
//! +-------------------+
//! ```
//!
//! This module provides:
//! - `ENFCAPTelemetryHeader`: Zero-copy header structure (64 bytes)
//! - `ENFCAPTelemetryIndex`: O(1) random frame access index
//! - `ENFCAPTelemetryReader`: Memory-mapped file reader
//! - `ENFCAPTelemetryWriter`: Buffered file writer
//! - `TelemetryFrameData`: Complete frame with vehicle state, entities, road points

use bytemuck::{Pod, Zeroable};
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use memmap2::{Mmap, MmapOptions};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// Constants
// ============================================================================

/// Magic bytes for ENFCAP telemetry format: "ENFCAP01"
pub const ENFCAP_TELEMETRY_MAGIC: [u8; 8] = [0x45, 0x4E, 0x46, 0x43, 0x41, 0x50, 0x30, 0x31];

/// Current format version
pub const ENFCAP_TELEMETRY_VERSION: u32 = 1;

/// Fixed header size in bytes
pub const HEADER_SIZE: usize = 64;

/// Size of each index entry (offset: u64)
pub const INDEX_ENTRY_SIZE: usize = 8;

/// Fixed portion of frame record (excluding variable-length data)
pub const FRAME_BASE_SIZE: usize = 102;

/// Size of each entity record in bytes (type:4 + pos:12 = 16)
pub const ENTITY_RECORD_SIZE: usize = 16;

/// Size of each road point record in bytes (pos:12)
pub const ROAD_POINT_SIZE: usize = 12;

// ============================================================================
// Header Flags
// ============================================================================

/// Header flags indicating file-level features
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeaderFlags(pub u32);

impl HeaderFlags {
    pub const COMPRESSED: u32 = 1;
    pub const HAS_SCREENSHOTS: u32 = 2;
    pub const HAS_DEPTH: u32 = 4;
    pub const HAS_AUDIO: u32 = 8;
    pub const ANCHOR_FRAMES: u32 = 16;

    pub fn contains(&self, flag: u32) -> bool {
        (self.0 & flag) != 0
    }

    pub fn set(&mut self, flag: u32) {
        self.0 |= flag;
    }
}

/// Per-frame flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FrameFlags(pub u16);

impl FrameFlags {
    pub const ANCHOR: u16 = 1;
    pub const COLLISION: u16 = 2;
    pub const MANUAL_CONTROL: u16 = 4;
    pub const SCENE_CHANGE: u16 = 8;

    pub fn contains(&self, flag: u16) -> bool {
        (self.0 & flag) != 0
    }

    pub fn set(&mut self, flag: u16) {
        self.0 |= flag;
    }

    pub fn is_anchor(&self) -> bool {
        self.contains(Self::ANCHOR)
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during ENFCAP telemetry operations
#[derive(Error, Debug)]
pub enum ENFCAPTelemetryError {
    #[error("Invalid magic bytes: expected ENFCAP01")]
    InvalidMagic,

    #[error("Unsupported version: {0} (expected {ENFCAP_TELEMETRY_VERSION})")]
    UnsupportedVersion(u32),

    #[error("Frame index {0} out of bounds (max: {1})")]
    FrameOutOfBounds(u32, u32),

    #[error("Corrupted frame data at offset {0}")]
    CorruptedFrame(u64),

    #[error("Invalid index table")]
    InvalidIndexTable,

    #[error("File too small: {0} bytes (minimum: {HEADER_SIZE})")]
    FileTooSmall(usize),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("Memory mapping error: {0}")]
    Mmap(String),
}

pub type Result<T> = std::result::Result<T, ENFCAPTelemetryError>;

// ============================================================================
// ENFCAPTelemetryHeader - Zero-Copy Header Structure
// ============================================================================

/// ENFCAP telemetry file header (64 bytes)
///
/// Layout:
/// - Magic: 8 bytes ("ENFCAP01")
/// - Version: 4 bytes
/// - Frame count: 4 bytes
/// - Start timestamp: 8 bytes (f64 as 2x f32)
/// - Flags: 4 bytes
/// - Reserved: 36 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ENFCAPTelemetryHeader {
    /// Magic bytes: "ENFCAP01"
    pub magic: [u8; 8],
    /// Format version (currently 1)
    pub version: u32,
    /// Total number of frames in the file
    pub frame_count: u32,
    /// Session start timestamp (milliseconds, stored as f64 via 2x f32)
    pub start_timestamp_lo: f32,
    pub start_timestamp_hi: f32,
    /// Header flags (see HeaderFlags)
    pub flags: u32,
    /// Reserved for future use
    pub reserved: [u8; 36],
}

impl ENFCAPTelemetryHeader {
    /// Create a new header with default values
    pub fn new(frame_count: u32, start_timestamp: f64, flags: HeaderFlags) -> Self {
        let mut header = Self::zeroed();
        header.magic = ENFCAP_TELEMETRY_MAGIC;
        header.version = ENFCAP_TELEMETRY_VERSION;
        header.frame_count = frame_count;
        header.start_timestamp_lo = start_timestamp as f32;
        header.start_timestamp_hi = 0.0; // For very large timestamps
        header.flags = flags.0;
        header
    }

    /// Validate the header magic and version
    pub fn validate(&self) -> Result<()> {
        if self.magic != ENFCAP_TELEMETRY_MAGIC {
            return Err(ENFCAPTelemetryError::InvalidMagic);
        }
        if self.version != ENFCAP_TELEMETRY_VERSION {
            return Err(ENFCAPTelemetryError::UnsupportedVersion(self.version));
        }
        Ok(())
    }

    /// Get the start timestamp as f64
    pub fn start_timestamp(&self) -> f64 {
        self.start_timestamp_lo as f64
    }

    /// Get header flags
    pub fn header_flags(&self) -> HeaderFlags {
        HeaderFlags(self.flags)
    }
}

impl Default for ENFCAPTelemetryHeader {
    fn default() -> Self {
        Self::new(0, 0.0, HeaderFlags(HeaderFlags::ANCHOR_FRAMES))
    }
}

// Ensure header size is exactly 64 bytes
const _: () = assert!(std::mem::size_of::<ENFCAPTelemetryHeader>() == HEADER_SIZE);

// ============================================================================
// Index Table
// ============================================================================

/// Index entry for a single frame
#[derive(Debug, Clone, Copy, Default)]
pub struct IndexEntry {
    /// Byte offset of frame data from start of file
    pub offset: u64,
}

/// Index table for O(1) random frame access
#[derive(Debug, Default)]
pub struct ENFCAPTelemetryIndex {
    entries: Vec<IndexEntry>,
}

impl ENFCAPTelemetryIndex {
    /// Create an empty index with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Create index from raw bytes
    pub fn from_bytes(data: &[u8], frame_count: u32) -> Result<Self> {
        let expected_size = frame_count as usize * INDEX_ENTRY_SIZE;
        if data.len() < expected_size {
            return Err(ENFCAPTelemetryError::InvalidIndexTable);
        }

        let entries: Vec<IndexEntry> = data[..expected_size]
            .chunks_exact(INDEX_ENTRY_SIZE)
            .map(|chunk| {
                let offset = u64::from_le_bytes(chunk.try_into().unwrap());
                IndexEntry { offset }
            })
            .collect();

        Ok(Self { entries })
    }

    /// Get the offset for a specific frame
    #[inline]
    pub fn get_offset(&self, frame_index: u32) -> Option<u64> {
        self.entries.get(frame_index as usize).map(|e| e.offset)
    }

    /// Get the number of indexed frames
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a new frame offset to the index
    pub fn push(&mut self, offset: u64) {
        self.entries.push(IndexEntry { offset });
    }

    /// Serialize the index to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.entries.len() * INDEX_ENTRY_SIZE);
        for entry in &self.entries {
            bytes.extend_from_slice(&entry.offset.to_le_bytes());
        }
        bytes
    }
}

// ============================================================================
// Scene Entity Data
// ============================================================================

/// Entity type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
#[pyclass]
pub enum EntityType {
    #[default]
    Vehicle = 0,
    Pedestrian = 1,
    Cyclist = 2,
    Animal = 3,
    TrafficSign = 4,
    TrafficLight = 5,
    Obstacle = 6,
    Other = 7,
}

impl From<u32> for EntityType {
    fn from(value: u32) -> Self {
        match value {
            0 => EntityType::Vehicle,
            1 => EntityType::Pedestrian,
            2 => EntityType::Cyclist,
            3 => EntityType::Animal,
            4 => EntityType::TrafficSign,
            5 => EntityType::TrafficLight,
            6 => EntityType::Obstacle,
            _ => EntityType::Other,
        }
    }
}

/// Compact representation of a scene entity
#[derive(Debug, Clone, Copy, Default)]
#[pyclass]
pub struct SceneEntity {
    /// Entity classification
    #[pyo3(get)]
    pub entity_type: u32,
    /// World position (x, y, z) in meters
    #[pyo3(get)]
    pub position: [f32; 3],
}

impl SceneEntity {
    /// Parse entity from raw bytes (16 bytes)
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= ENTITY_RECORD_SIZE);

        let entity_type = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let x = f32::from_le_bytes(data[4..8].try_into().unwrap());
        let y = f32::from_le_bytes(data[8..12].try_into().unwrap());
        let z = f32::from_le_bytes(data[12..16].try_into().unwrap());

        Self {
            entity_type,
            position: [x, y, z],
        }
    }

    /// Serialize entity to bytes
    pub fn to_bytes(&self) -> [u8; ENTITY_RECORD_SIZE] {
        let mut bytes = [0u8; ENTITY_RECORD_SIZE];
        bytes[0..4].copy_from_slice(&self.entity_type.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.position[0].to_le_bytes());
        bytes[8..12].copy_from_slice(&self.position[1].to_le_bytes());
        bytes[12..16].copy_from_slice(&self.position[2].to_le_bytes());
        bytes
    }
}

// ============================================================================
// Road Point Data
// ============================================================================

/// Compact road topology point
#[derive(Debug, Clone, Copy, Default)]
#[pyclass]
pub struct RoadPoint {
    /// World position (x, y, z) in meters
    #[pyo3(get)]
    pub position: [f32; 3],
}

impl RoadPoint {
    /// Parse road point from raw bytes (12 bytes)
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= ROAD_POINT_SIZE);

        let x = f32::from_le_bytes(data[0..4].try_into().unwrap());
        let y = f32::from_le_bytes(data[4..8].try_into().unwrap());
        let z = f32::from_le_bytes(data[8..12].try_into().unwrap());

        Self {
            position: [x, y, z],
        }
    }

    /// Serialize road point to bytes
    pub fn to_bytes(&self) -> [u8; ROAD_POINT_SIZE] {
        let mut bytes = [0u8; ROAD_POINT_SIZE];
        bytes[0..4].copy_from_slice(&self.position[0].to_le_bytes());
        bytes[4..8].copy_from_slice(&self.position[1].to_le_bytes());
        bytes[8..12].copy_from_slice(&self.position[2].to_le_bytes());
        bytes
    }
}

// ============================================================================
// Vehicle State
// ============================================================================

/// Complete vehicle state for a single frame (10 floats = 40 bytes)
#[derive(Debug, Clone, Copy, Default)]
#[pyclass]
pub struct VehicleState {
    /// Speed in km/h
    #[pyo3(get)]
    pub speed_kmh: f32,
    /// Steering angle (-1.0 to 1.0)
    #[pyo3(get)]
    pub steering: f32,
    /// Throttle position (0.0 to 1.0)
    #[pyo3(get)]
    pub throttle: f32,
    /// Brake position (0.0 to 1.0)
    #[pyo3(get)]
    pub brake: f32,
    /// Clutch position (0.0 to 1.0)
    #[pyo3(get)]
    pub clutch: f32,
    /// Current gear
    #[pyo3(get)]
    pub gear: f32,
    /// Engine RPM
    #[pyo3(get)]
    pub engine_rpm: f32,
    /// Longitudinal acceleration (m/s^2)
    #[pyo3(get)]
    pub acceleration: f32,
    /// Yaw rate (rad/s)
    #[pyo3(get)]
    pub yaw_rate: f32,
    /// Slip angle (radians)
    #[pyo3(get)]
    pub slip_angle: f32,
}

impl VehicleState {
    /// Parse vehicle state from raw bytes (40 bytes = 10 * f32)
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= 40);

        Self {
            speed_kmh: f32::from_le_bytes(data[0..4].try_into().unwrap()),
            steering: f32::from_le_bytes(data[4..8].try_into().unwrap()),
            throttle: f32::from_le_bytes(data[8..12].try_into().unwrap()),
            brake: f32::from_le_bytes(data[12..16].try_into().unwrap()),
            clutch: f32::from_le_bytes(data[16..20].try_into().unwrap()),
            gear: f32::from_le_bytes(data[20..24].try_into().unwrap()),
            engine_rpm: f32::from_le_bytes(data[24..28].try_into().unwrap()),
            acceleration: f32::from_le_bytes(data[28..32].try_into().unwrap()),
            yaw_rate: f32::from_le_bytes(data[32..36].try_into().unwrap()),
            slip_angle: f32::from_le_bytes(data[36..40].try_into().unwrap()),
        }
    }

    /// Serialize vehicle state to bytes
    pub fn to_bytes(&self) -> [u8; 40] {
        let mut bytes = [0u8; 40];
        bytes[0..4].copy_from_slice(&self.speed_kmh.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.steering.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.throttle.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.brake.to_le_bytes());
        bytes[16..20].copy_from_slice(&self.clutch.to_le_bytes());
        bytes[20..24].copy_from_slice(&self.gear.to_le_bytes());
        bytes[24..28].copy_from_slice(&self.engine_rpm.to_le_bytes());
        bytes[28..32].copy_from_slice(&self.acceleration.to_le_bytes());
        bytes[32..36].copy_from_slice(&self.yaw_rate.to_le_bytes());
        bytes[36..40].copy_from_slice(&self.slip_angle.to_le_bytes());
        bytes
    }
}

// ============================================================================
// Ego Transform (3x4 matrix)
// ============================================================================

/// 3x4 transformation matrix for ego vehicle pose (48 bytes = 12 * f32)
#[derive(Debug, Clone, Copy, Default)]
#[pyclass]
pub struct EgoTransformMatrix {
    /// 3x4 transformation matrix (12 floats, row-major)
    /// Row 0: Right vector (x, y, z)
    /// Row 1: Up vector (x, y, z)
    /// Row 2: Forward vector (x, y, z)
    /// Row 3: Position (x, y, z)
    #[pyo3(get)]
    pub matrix: [f32; 12],
}

#[pymethods]
impl EgoTransformMatrix {
    /// Get the position component (last row)
    fn position(&self) -> [f32; 3] {
        [self.matrix[9], self.matrix[10], self.matrix[11]]
    }

    /// Get the forward direction (third row)
    fn forward(&self) -> [f32; 3] {
        [self.matrix[6], self.matrix[7], self.matrix[8]]
    }

    /// Get the right direction (first row)
    fn right(&self) -> [f32; 3] {
        [self.matrix[0], self.matrix[1], self.matrix[2]]
    }

    /// Get the up direction (second row)
    fn up(&self) -> [f32; 3] {
        [self.matrix[3], self.matrix[4], self.matrix[5]]
    }
}

impl EgoTransformMatrix {
    /// Parse transform from raw bytes (48 bytes = 12 * f32)
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= 48);

        let mut matrix = [0.0f32; 12];
        for i in 0..12 {
            matrix[i] = f32::from_le_bytes(data[i * 4..(i + 1) * 4].try_into().unwrap());
        }

        Self { matrix }
    }

    /// Serialize transform to bytes
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut bytes = [0u8; 48];
        for i in 0..12 {
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&self.matrix[i].to_le_bytes());
        }
        bytes
    }
}

// ============================================================================
// TelemetryFrameData - Complete Frame Record
// ============================================================================

/// Complete telemetry frame data container
#[derive(Debug, Clone, Default)]
#[pyclass]
pub struct TelemetryFrameData {
    /// Unique frame identifier within session
    #[pyo3(get)]
    pub frame_id: u32,
    /// Timestamp in milliseconds since session start
    #[pyo3(get)]
    pub timestamp: f32,
    /// Ego vehicle transformation matrix
    #[pyo3(get)]
    pub ego_transform: EgoTransformMatrix,
    /// Vehicle dynamics state
    #[pyo3(get)]
    pub vehicle_state: VehicleState,
    /// Scene entities detected this frame
    #[pyo3(get)]
    pub scene_entities: Vec<SceneEntity>,
    /// Road topology points
    #[pyo3(get)]
    pub road_points: Vec<RoadPoint>,
    /// Frame-level flags
    #[pyo3(get)]
    pub flags: u16,
}

#[pymethods]
impl TelemetryFrameData {
    /// Check if this is an anchor (keyframe)
    fn is_anchor(&self) -> bool {
        (self.flags & FrameFlags::ANCHOR) != 0
    }

    /// Check if a collision was detected
    fn has_collision(&self) -> bool {
        (self.flags & FrameFlags::COLLISION) != 0
    }

    /// Check if this frame has manual (human) control
    fn is_manual_control(&self) -> bool {
        (self.flags & FrameFlags::MANUAL_CONTROL) != 0
    }

    fn __repr__(&self) -> String {
        format!(
            "TelemetryFrameData(id={}, t={:.1}ms, entities={}, road_pts={})",
            self.frame_id,
            self.timestamp,
            self.scene_entities.len(),
            self.road_points.len()
        )
    }
}

impl TelemetryFrameData {
    /// Parse a complete frame from raw bytes
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < FRAME_BASE_SIZE {
            return Err(ENFCAPTelemetryError::CorruptedFrame(0));
        }

        let mut offset = 0;

        // Frame ID (4 bytes)
        let frame_id = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        // Timestamp (4 bytes)
        let timestamp = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        // Ego Transform (48 bytes)
        let ego_transform = EgoTransformMatrix::from_bytes(&data[offset..offset + 48]);
        offset += 48;

        // Vehicle State (40 bytes)
        let vehicle_state = VehicleState::from_bytes(&data[offset..offset + 40]);
        offset += 40;

        // Scene Entity Count (2 bytes)
        let entity_count =
            u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;

        // Scene Entities (entity_count * 16 bytes)
        let entities_size = entity_count * ENTITY_RECORD_SIZE;
        if data.len() < offset + entities_size {
            return Err(ENFCAPTelemetryError::CorruptedFrame(offset as u64));
        }

        let mut scene_entities = Vec::with_capacity(entity_count);
        for i in 0..entity_count {
            let entity_offset = offset + i * ENTITY_RECORD_SIZE;
            scene_entities.push(SceneEntity::from_bytes(&data[entity_offset..]));
        }
        offset += entities_size;

        // Road Point Count (2 bytes)
        let road_count = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;

        // Road Points (road_count * 12 bytes)
        let roads_size = road_count * ROAD_POINT_SIZE;
        if data.len() < offset + roads_size {
            return Err(ENFCAPTelemetryError::CorruptedFrame(offset as u64));
        }

        let mut road_points = Vec::with_capacity(road_count);
        for i in 0..road_count {
            let road_offset = offset + i * ROAD_POINT_SIZE;
            road_points.push(RoadPoint::from_bytes(&data[road_offset..]));
        }
        offset += roads_size;

        // Frame Flags (2 bytes)
        let flags = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
        offset += 2;

        Ok((
            Self {
                frame_id,
                timestamp,
                ego_transform,
                vehicle_state,
                scene_entities,
                road_points,
                flags,
            },
            offset,
        ))
    }

    /// Serialize frame to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let size = self.serialized_size();
        let mut bytes = Vec::with_capacity(size);

        // Frame ID
        bytes.extend_from_slice(&self.frame_id.to_le_bytes());

        // Timestamp
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());

        // Ego Transform
        bytes.extend_from_slice(&self.ego_transform.to_bytes());

        // Vehicle State
        bytes.extend_from_slice(&self.vehicle_state.to_bytes());

        // Scene Entity Count
        let entity_count = self.scene_entities.len().min(u16::MAX as usize) as u16;
        bytes.extend_from_slice(&entity_count.to_le_bytes());

        // Scene Entities
        for entity in &self.scene_entities[..entity_count as usize] {
            bytes.extend_from_slice(&entity.to_bytes());
        }

        // Road Point Count
        let road_count = self.road_points.len().min(u16::MAX as usize) as u16;
        bytes.extend_from_slice(&road_count.to_le_bytes());

        // Road Points
        for point in &self.road_points[..road_count as usize] {
            bytes.extend_from_slice(&point.to_bytes());
        }

        // Frame Flags
        bytes.extend_from_slice(&self.flags.to_le_bytes());

        bytes
    }

    /// Calculate the serialized size of this frame
    pub fn serialized_size(&self) -> usize {
        FRAME_BASE_SIZE
            + self.scene_entities.len().min(u16::MAX as usize) * ENTITY_RECORD_SIZE
            + self.road_points.len().min(u16::MAX as usize) * ROAD_POINT_SIZE
    }
}

// ============================================================================
// ENFCAPTelemetryReader - Memory-Mapped File Reader
// ============================================================================

/// High-performance ENFCAP telemetry file reader with memory-mapped access
#[pyclass]
pub struct ENFCAPTelemetryReader {
    /// Memory-mapped file data
    mmap: Arc<Mmap>,
    /// Parsed file header
    header: ENFCAPTelemetryHeader,
    /// Index table for random access
    index: ENFCAPTelemetryIndex,
    /// File path for error reporting
    path: String,
}

#[pymethods]
impl ENFCAPTelemetryReader {
    /// Open an ENFCAP telemetry file for reading
    #[new]
    fn py_new(path: &str) -> PyResult<Self> {
        Self::open(path).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Get the number of frames in the file
    fn frame_count(&self) -> u32 {
        self.header.frame_count
    }

    /// Get the session start timestamp
    fn start_timestamp(&self) -> f64 {
        self.header.start_timestamp()
    }

    /// Read a single frame by index (O(1) random access)
    fn read_frame(&self, frame_index: u32) -> PyResult<TelemetryFrameData> {
        self.read_frame_inner(frame_index)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Read multiple frames by indices
    fn read_frames(&self, frame_indices: Vec<u32>) -> PyResult<Vec<TelemetryFrameData>> {
        frame_indices
            .iter()
            .map(|&idx| self.read_frame(idx))
            .collect()
    }

    /// Read a range of consecutive frames
    fn read_range(&self, start: u32, end: u32) -> PyResult<Vec<TelemetryFrameData>> {
        self.read_frames_parallel_inner(start..end)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Get file info as a dictionary
    fn info(&self) -> HashMap<String, String> {
        let mut info = HashMap::new();
        info.insert("path".to_string(), self.path.clone());
        info.insert("version".to_string(), self.header.version.to_string());
        info.insert("frame_count".to_string(), self.header.frame_count.to_string());
        info.insert(
            "start_timestamp".to_string(),
            self.header.start_timestamp().to_string(),
        );
        info.insert("flags".to_string(), self.header.flags.to_string());
        info
    }

    fn __repr__(&self) -> String {
        format!(
            "ENFCAPTelemetryReader({}, {} frames)",
            self.path, self.header.frame_count
        )
    }

    fn __len__(&self) -> usize {
        self.header.frame_count as usize
    }
}

impl ENFCAPTelemetryReader {
    /// Open an ENFCAP telemetry file for reading
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(&path)?;
        let metadata = file.metadata()?;

        if metadata.len() < HEADER_SIZE as u64 {
            return Err(ENFCAPTelemetryError::FileTooSmall(metadata.len() as usize));
        }

        // Memory-map the file
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| ENFCAPTelemetryError::Mmap(e.to_string()))?
        };

        // Parse header
        let header: ENFCAPTelemetryHeader = *bytemuck::from_bytes(&mmap[..HEADER_SIZE]);
        header.validate()?;

        // Calculate index table location (at end of file)
        let index_size = header.frame_count as usize * INDEX_ENTRY_SIZE;
        let index_offset = mmap.len() - index_size;

        // Parse index table
        let index = ENFCAPTelemetryIndex::from_bytes(&mmap[index_offset..], header.frame_count)?;

        Ok(Self {
            mmap: Arc::new(mmap),
            header,
            index,
            path: path_str,
        })
    }

    /// Read a single frame by index
    fn read_frame_inner(&self, frame_index: u32) -> Result<TelemetryFrameData> {
        if frame_index >= self.header.frame_count {
            return Err(ENFCAPTelemetryError::FrameOutOfBounds(
                frame_index,
                self.header.frame_count,
            ));
        }

        let offset = self
            .index
            .get_offset(frame_index)
            .ok_or(ENFCAPTelemetryError::InvalidIndexTable)? as usize;

        let (frame, _) = TelemetryFrameData::from_bytes(&self.mmap[offset..])?;
        Ok(frame)
    }

    /// Read multiple frames in parallel using rayon
    fn read_frames_parallel_inner(
        &self,
        range: std::ops::Range<u32>,
    ) -> Result<Vec<TelemetryFrameData>> {
        let start = range.start;
        let end = range.end.min(self.header.frame_count);

        if start >= end {
            return Ok(Vec::new());
        }

        let mmap = &self.mmap;
        let index = &self.index;

        let frames: Result<Vec<TelemetryFrameData>> = (start..end)
            .into_par_iter()
            .map(|idx| {
                let offset = index
                    .get_offset(idx)
                    .ok_or(ENFCAPTelemetryError::InvalidIndexTable)?
                    as usize;
                TelemetryFrameData::from_bytes(&mmap[offset..]).map(|(f, _)| f)
            })
            .collect();

        frames
    }

    /// Iterate over all frames
    pub fn iter_frames(&self) -> impl Iterator<Item = Result<TelemetryFrameData>> + '_ {
        (0..self.header.frame_count).map(move |i| self.read_frame_inner(i))
    }
}

// ============================================================================
// ENFCAPTelemetryWriter - File Writer
// ============================================================================

/// ENFCAP telemetry file writer with buffered output
#[pyclass]
pub struct ENFCAPTelemetryWriter {
    /// Buffered file writer
    writer: BufWriter<File>,
    /// File path
    path: String,
    /// Frame offsets for index table
    index: ENFCAPTelemetryIndex,
    /// Session start timestamp
    start_timestamp: f64,
    /// Header flags
    flags: HeaderFlags,
    /// Current write position
    position: u64,
    /// Anchor frame interval
    anchor_interval: u32,
    /// Whether the file has been finalized
    finalized: bool,
}

#[pymethods]
impl ENFCAPTelemetryWriter {
    /// Create a new ENFCAP telemetry file for writing
    #[new]
    #[pyo3(signature = (path, anchor_interval = 100))]
    fn py_new(path: &str, anchor_interval: u32) -> PyResult<Self> {
        Self::create(path, anchor_interval)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Set the session start timestamp
    fn set_start_timestamp(&mut self, timestamp: f64) {
        self.start_timestamp = timestamp;
    }

    /// Write a frame to the file
    fn write_frame(&mut self, frame: &TelemetryFrameData) -> PyResult<()> {
        self.write_frame_inner(frame)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Get the current frame count
    fn frame_count(&self) -> u32 {
        self.index.len() as u32
    }

    /// Finalize the file (write index table and update header)
    fn finalize(&mut self) -> PyResult<()> {
        self.finalize_inner()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        self.finalized = true;
        Ok(())
    }

    /// Get the file path
    fn path(&self) -> &str {
        &self.path
    }
}

impl ENFCAPTelemetryWriter {
    /// Create a new ENFCAP telemetry file for writing
    pub fn create<P: AsRef<Path>>(path: P, anchor_interval: u32) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        let mut writer = BufWriter::with_capacity(64 * 1024, file);

        // Write placeholder header
        let header = ENFCAPTelemetryHeader::default();
        writer.write_all(bytemuck::bytes_of(&header))?;

        Ok(Self {
            writer,
            path: path_str,
            index: ENFCAPTelemetryIndex::with_capacity(10000),
            start_timestamp: 0.0,
            flags: HeaderFlags(HeaderFlags::ANCHOR_FRAMES),
            position: HEADER_SIZE as u64,
            anchor_interval,
            finalized: false,
        })
    }

    /// Write a frame to the file
    fn write_frame_inner(&mut self, frame: &TelemetryFrameData) -> Result<()> {
        // Record offset for this frame
        self.index.push(self.position);

        // Check if this should be an anchor frame
        let frame_count = self.index.len() as u32;
        let mut frame_data = frame.clone();
        if frame_count % self.anchor_interval == 0 {
            frame_data.flags |= FrameFlags::ANCHOR;
        }

        // Serialize and write frame
        let bytes = frame_data.to_bytes();
        self.writer.write_all(&bytes)?;
        self.position += bytes.len() as u64;

        Ok(())
    }

    /// Finalize the file
    fn finalize_inner(&mut self) -> Result<()> {
        // Write index table
        let index_bytes = self.index.to_bytes();
        self.writer.write_all(&index_bytes)?;

        // Flush all buffered data
        self.writer.flush()?;

        // Update header with final values
        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;

        let header = ENFCAPTelemetryHeader::new(
            self.index.len() as u32,
            self.start_timestamp,
            self.flags,
        );
        file.write_all(bytemuck::bytes_of(&header))?;

        file.flush()?;

        Ok(())
    }
}

impl Drop for ENFCAPTelemetryWriter {
    fn drop(&mut self) {
        if !self.finalized && !self.index.is_empty() {
            // Attempt to finalize on drop, but ignore errors
            let _ = self.finalize_inner();
        }
    }
}

// ============================================================================
// Validation Report
// ============================================================================

/// Validation report for file integrity checks
#[derive(Debug)]
#[pyclass]
pub struct ValidationReport {
    #[pyo3(get)]
    pub frame_count: u32,
    #[pyo3(get)]
    pub valid_frames: u32,
    #[pyo3(get)]
    pub total_entities: usize,
    #[pyo3(get)]
    pub total_road_points: usize,
    #[pyo3(get)]
    pub anchor_count: usize,
}

#[pymethods]
impl ValidationReport {
    fn is_valid(&self) -> bool {
        self.valid_frames == self.frame_count
    }

    fn __repr__(&self) -> String {
        format!(
            "ValidationReport(frames={}/{}, entities={}, road_pts={}, anchors={})",
            self.valid_frames,
            self.frame_count,
            self.total_entities,
            self.total_road_points,
            self.anchor_count
        )
    }
}

/// Validate an ENFCAP telemetry file
pub fn validate_file<P: AsRef<Path>>(path: P) -> Result<ValidationReport> {
    let reader = ENFCAPTelemetryReader::open(path)?;

    let mut report = ValidationReport {
        frame_count: reader.header.frame_count,
        valid_frames: 0,
        total_entities: 0,
        total_road_points: 0,
        anchor_count: 0,
    };

    for frame_result in reader.iter_frames() {
        if let Ok(frame) = frame_result {
            report.valid_frames += 1;
            report.total_entities += frame.scene_entities.len();
            report.total_road_points += frame.road_points.len();
            if frame.is_anchor() {
                report.anchor_count += 1;
            }
        }
    }

    Ok(report)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_header_size() {
        assert_eq!(std::mem::size_of::<ENFCAPTelemetryHeader>(), HEADER_SIZE);
    }

    #[test]
    fn test_header_roundtrip() {
        let header = ENFCAPTelemetryHeader::new(100, 12345.678, HeaderFlags(HeaderFlags::COMPRESSED));
        assert_eq!(header.magic, ENFCAP_TELEMETRY_MAGIC);
        assert_eq!(header.version, ENFCAP_TELEMETRY_VERSION);
        assert_eq!(header.frame_count, 100);
        assert!(header.header_flags().contains(HeaderFlags::COMPRESSED));
    }

    #[test]
    fn test_frame_roundtrip() {
        let frame = TelemetryFrameData {
            frame_id: 42,
            timestamp: 1234.5,
            ego_transform: EgoTransformMatrix {
                matrix: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 10.0, 20.0, 30.0],
            },
            vehicle_state: VehicleState {
                speed_kmh: 60.0,
                steering: 0.1,
                throttle: 0.5,
                brake: 0.0,
                clutch: 0.0,
                gear: 3.0,
                engine_rpm: 3000.0,
                acceleration: 1.5,
                yaw_rate: 0.01,
                slip_angle: 0.02,
            },
            scene_entities: vec![SceneEntity {
                entity_type: 0,
                position: [100.0, 0.0, 50.0],
            }],
            road_points: vec![
                RoadPoint {
                    position: [0.0, 0.0, 0.0],
                },
                RoadPoint {
                    position: [10.0, 0.0, 10.0],
                },
            ],
            flags: FrameFlags::ANCHOR,
        };

        let bytes = frame.to_bytes();
        let (parsed, size) = TelemetryFrameData::from_bytes(&bytes).unwrap();

        assert_eq!(size, bytes.len());
        assert_eq!(parsed.frame_id, frame.frame_id);
        assert_eq!(parsed.timestamp, frame.timestamp);
        assert_eq!(parsed.scene_entities.len(), 1);
        assert_eq!(parsed.road_points.len(), 2);
        assert!(parsed.is_anchor());
    }

    #[test]
    fn test_write_and_read() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Write some frames
        {
            let mut writer = ENFCAPTelemetryWriter::create(&path, 100).unwrap();
            writer.set_start_timestamp(1000.0);

            for i in 0..10 {
                let frame = TelemetryFrameData {
                    frame_id: i,
                    timestamp: i as f32 * 100.0,
                    ..Default::default()
                };
                writer.write_frame_inner(&frame).unwrap();
            }

            writer.finalize_inner().unwrap();
        }

        // Read them back
        {
            let reader = ENFCAPTelemetryReader::open(&path).unwrap();
            assert_eq!(reader.frame_count(), 10);

            for i in 0..10 {
                let frame = reader.read_frame_inner(i).unwrap();
                assert_eq!(frame.frame_id, i);
                assert_eq!(frame.timestamp, i as f32 * 100.0);
            }
        }
    }

    #[test]
    fn test_index_operations() {
        let mut index = ENFCAPTelemetryIndex::with_capacity(10);

        for i in 0..10 {
            index.push(i as u64 * 1000);
        }

        assert_eq!(index.len(), 10);
        assert_eq!(index.get_offset(5), Some(5000));
        assert_eq!(index.get_offset(100), None);

        let bytes = index.to_bytes();
        let parsed = ENFCAPTelemetryIndex::from_bytes(&bytes, 10).unwrap();
        assert_eq!(parsed.len(), 10);
        assert_eq!(parsed.get_offset(5), Some(5000));
    }
}
