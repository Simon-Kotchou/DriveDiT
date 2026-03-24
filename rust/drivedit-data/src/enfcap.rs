//! ENFCAP Binary Format Parser
//!
//! High-performance Rust implementation of the ENFCAP (ENFusion CAPture) binary format
//! for autonomous driving data capture. Designed for zero-copy memory-mapped file access
//! with O(1) random frame access.
//!
//! # File Format Specification (ENFCAP v1)
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
//! # Example Usage
//!
//! ```rust,ignore
//! use drivedit_data::enfcap::{ENFCAPReader, ENFCAPWriter, FrameData};
//!
//! // Reading frames with random access
//! let reader = ENFCAPReader::open("session_001.enfcap")?;
//! let frame = reader.read_frame(42)?;  // O(1) random access
//! println!("Frame {} at timestamp {}", frame.frame_id, frame.timestamp);
//!
//! // Parallel frame decoding
//! let frames: Vec<FrameData> = reader.read_frames_parallel(0..100)?;
//!
//! // Writing new capture files
//! let mut writer = ENFCAPWriter::create("session_002.enfcap")?;
//! writer.write_frame(&frame)?;
//! writer.finalize()?;
//! ```

use bytemuck::{Pod, Zeroable};
use memmap2::{Mmap, MmapMut, MmapOptions};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;

// ============================================================================
// Constants
// ============================================================================

/// Magic bytes for ENFCAP format: "ENFCAP01"
pub const ENFCAP_MAGIC: [u8; 8] = [0x45, 0x4E, 0x46, 0x43, 0x41, 0x50, 0x30, 0x31];

/// Current format version
pub const ENFCAP_VERSION: u32 = 1;

/// Fixed header size in bytes
pub const ENFCAP_HEADER_SIZE: usize = 64;

/// Size of each index entry (offset: u64)
pub const ENFCAP_INDEX_ENTRY_SIZE: usize = 8;

/// Fixed portion of frame record (excluding variable-length data)
pub const FRAME_BASE_SIZE: usize = 102;

/// Size of each entity record in bytes
pub const ENTITY_RECORD_SIZE: usize = 16;

/// Size of each road point record in bytes
pub const ROAD_POINT_SIZE: usize = 12;

// ============================================================================
// Header Flags
// ============================================================================

bitflags::bitflags! {
    /// Header flags indicating file-level features
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct HeaderFlags: u32 {
        /// Data is compressed (LZ4)
        const COMPRESSED = 1;
        /// External screenshot sync available
        const HAS_SCREENSHOTS = 2;
        /// Depth data included
        const HAS_DEPTH = 4;
        /// Audio data included
        const HAS_AUDIO = 8;
        /// Contains anchor frames for random access
        const ANCHOR_FRAMES = 16;
    }
}

bitflags::bitflags! {
    /// Per-frame flags indicating frame-level features
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct FrameFlags: u16 {
        /// This frame is an anchor (keyframe) for random access
        const ANCHOR = 1;
        /// Collision detected this frame
        const COLLISION = 2;
        /// Human driving (not AI)
        const MANUAL_CONTROL = 4;
        /// Significant scene change
        const SCENE_CHANGE = 8;
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during ENFCAP operations
#[derive(Error, Debug)]
pub enum ENFCAPError {
    #[error("Invalid magic bytes: expected ENFCAP01")]
    InvalidMagic,

    #[error("Unsupported version: {0} (expected {ENFCAP_VERSION})")]
    UnsupportedVersion(u32),

    #[error("Frame index {0} out of bounds (max: {1})")]
    FrameOutOfBounds(u32, u32),

    #[error("Corrupted frame data at offset {0}")]
    CorruptedFrame(u64),

    #[error("Invalid index table")]
    InvalidIndexTable,

    #[error("File too small: {0} bytes (minimum: {ENFCAP_HEADER_SIZE})")]
    FileTooSmall(usize),

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("Memory mapping error: {0}")]
    Mmap(String),

    #[error("File not finalized - call finalize() before closing")]
    NotFinalized,
}

pub type Result<T> = std::result::Result<T, ENFCAPError>;

// ============================================================================
// ENFCAPHeader - Zero-Copy Header Structure
// ============================================================================

/// ENFCAP file header (64 bytes)
///
/// This struct is designed for zero-copy reads using bytemuck.
/// All fields are little-endian.
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ENFCAPHeader {
    /// Magic bytes: "ENFCAP01"
    pub magic: [u8; 8],
    /// Format version (currently 1)
    pub version: u32,
    /// Total number of frames in the file
    pub frame_count: u32,
    /// Session start timestamp (milliseconds, stored as f64)
    pub start_timestamp_lo: f32,
    pub start_timestamp_hi: f32,
    /// Header flags (see HeaderFlags)
    pub flags: u32,
    /// Reserved for future use
    pub reserved: [u8; 36],
}

impl ENFCAPHeader {
    /// Create a new header with default values
    pub fn new(frame_count: u32, start_timestamp: f64, flags: HeaderFlags) -> Self {
        let mut header = Self::zeroed();
        header.magic = ENFCAP_MAGIC;
        header.version = ENFCAP_VERSION;
        header.frame_count = frame_count;
        header.start_timestamp_lo = start_timestamp as f32;
        header.start_timestamp_hi = (start_timestamp / (f32::MAX as f64)) as f32;
        header.flags = flags.bits();
        header
    }

    /// Validate the header magic and version
    pub fn validate(&self) -> Result<()> {
        if self.magic != ENFCAP_MAGIC {
            return Err(ENFCAPError::InvalidMagic);
        }
        if self.version != ENFCAP_VERSION {
            return Err(ENFCAPError::UnsupportedVersion(self.version));
        }
        Ok(())
    }

    /// Get the start timestamp as f64
    pub fn start_timestamp(&self) -> f64 {
        self.start_timestamp_lo as f64 + (self.start_timestamp_hi as f64 * f32::MAX as f64)
    }

    /// Get header flags
    pub fn header_flags(&self) -> HeaderFlags {
        HeaderFlags::from_bits_truncate(self.flags)
    }
}

impl Default for ENFCAPHeader {
    fn default() -> Self {
        Self::new(0, 0.0, HeaderFlags::ANCHOR_FRAMES)
    }
}

// Ensure header size is exactly 64 bytes
const _: () = assert!(std::mem::size_of::<ENFCAPHeader>() == ENFCAP_HEADER_SIZE);

// ============================================================================
// ENFCAPIndex - Index Table for Random Access
// ============================================================================

/// Index entry for a single frame (8 bytes)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct IndexEntry {
    /// Byte offset of frame data from start of file
    pub offset: u64,
}

/// Index table for O(1) random frame access
#[derive(Debug)]
pub struct ENFCAPIndex {
    /// Frame offsets for random access
    entries: Vec<IndexEntry>,
}

impl ENFCAPIndex {
    /// Create an empty index with pre-allocated capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
        }
    }

    /// Create index from raw bytes (zero-copy when possible)
    pub fn from_bytes(data: &[u8], frame_count: u32) -> Result<Self> {
        let expected_size = frame_count as usize * ENFCAP_INDEX_ENTRY_SIZE;
        if data.len() < expected_size {
            return Err(ENFCAPError::InvalidIndexTable);
        }

        let entries: Vec<IndexEntry> = data[..expected_size]
            .chunks_exact(ENFCAP_INDEX_ENTRY_SIZE)
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
        let mut bytes = Vec::with_capacity(self.entries.len() * ENFCAP_INDEX_ENTRY_SIZE);
        for entry in &self.entries {
            bytes.extend_from_slice(&entry.offset.to_le_bytes());
        }
        bytes
    }

    /// Get a range of offsets for parallel processing
    pub fn get_range(&self, range: Range<u32>) -> &[IndexEntry] {
        let start = range.start as usize;
        let end = (range.end as usize).min(self.entries.len());
        &self.entries[start..end]
    }
}

// ============================================================================
// Scene Entity Data
// ============================================================================

/// Entity type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u32)]
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
pub struct SceneEntity {
    /// Entity classification
    pub entity_type: EntityType,
    /// World position (x, y, z) in meters
    pub position: [f32; 3],
    /// Facing direction (radians)
    pub yaw: f32,
    /// Speed (m/s)
    pub velocity: f32,
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
            entity_type: EntityType::from(entity_type),
            position: [x, y, z],
            yaw: 0.0,      // Not stored in current format
            velocity: 0.0, // Not stored in current format
        }
    }

    /// Serialize entity to bytes
    pub fn to_bytes(&self) -> [u8; ENTITY_RECORD_SIZE] {
        let mut bytes = [0u8; ENTITY_RECORD_SIZE];
        bytes[0..4].copy_from_slice(&(self.entity_type as u32).to_le_bytes());
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
pub struct RoadPoint {
    /// World position (x, y, z) in meters
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

/// Complete vehicle state for a single frame
#[derive(Debug, Clone, Copy, Default)]
pub struct VehicleState {
    /// Speed in km/h
    pub speed_kmh: f32,
    /// Steering angle (-1.0 to 1.0)
    pub steering: f32,
    /// Throttle position (0.0 to 1.0)
    pub throttle: f32,
    /// Brake position (0.0 to 1.0)
    pub brake: f32,
    /// Clutch position (0.0 to 1.0)
    pub clutch: f32,
    /// Current gear (-1 = reverse, 0 = neutral, 1+ = forward gears)
    pub gear: f32,
    /// Engine RPM
    pub engine_rpm: f32,
    /// Longitudinal acceleration (m/s^2)
    pub acceleration: f32,
    /// Yaw rate (rad/s)
    pub yaw_rate: f32,
    /// Slip angle (radians)
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
// Ego Transform
// ============================================================================

/// 3x4 transformation matrix for ego vehicle pose
///
/// Row-major layout:
/// - Row 0: Right vector (x, y, z)
/// - Row 1: Up vector (x, y, z)
/// - Row 2: Forward vector (x, y, z)
/// - Row 3: Position (x, y, z)
#[derive(Debug, Clone, Copy, Default)]
pub struct EgoTransform {
    /// 3x4 transformation matrix (12 floats, row-major)
    pub matrix: [f32; 12],
}

impl EgoTransform {
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

    /// Get the position component (last row)
    #[inline]
    pub fn position(&self) -> [f32; 3] {
        [self.matrix[9], self.matrix[10], self.matrix[11]]
    }

    /// Get the forward direction (third row)
    #[inline]
    pub fn forward(&self) -> [f32; 3] {
        [self.matrix[6], self.matrix[7], self.matrix[8]]
    }

    /// Get the right direction (first row)
    #[inline]
    pub fn right(&self) -> [f32; 3] {
        [self.matrix[0], self.matrix[1], self.matrix[2]]
    }

    /// Get the up direction (second row)
    #[inline]
    pub fn up(&self) -> [f32; 3] {
        [self.matrix[3], self.matrix[4], self.matrix[5]]
    }

    /// Convert to nalgebra Matrix4x4 (for full transform operations)
    #[cfg(feature = "nalgebra")]
    pub fn to_nalgebra(&self) -> nalgebra::Matrix4<f32> {
        nalgebra::Matrix4::new(
            self.matrix[0], self.matrix[1], self.matrix[2], 0.0,
            self.matrix[3], self.matrix[4], self.matrix[5], 0.0,
            self.matrix[6], self.matrix[7], self.matrix[8], 0.0,
            self.matrix[9], self.matrix[10], self.matrix[11], 1.0,
        )
    }
}

// ============================================================================
// FrameData - Complete Frame Record
// ============================================================================

/// Complete frame data container
#[derive(Debug, Clone, Default)]
pub struct FrameData {
    /// Unique frame identifier within session
    pub frame_id: u32,
    /// Timestamp in milliseconds since session start
    pub timestamp: f32,
    /// Ego vehicle transformation matrix
    pub ego_transform: EgoTransform,
    /// Vehicle dynamics state
    pub vehicle_state: VehicleState,
    /// Scene entities detected this frame
    pub scene_entities: Vec<SceneEntity>,
    /// Road topology points
    pub road_points: Vec<RoadPoint>,
    /// Frame-level flags
    pub flags: FrameFlags,
}

impl FrameData {
    /// Parse a complete frame from raw bytes
    ///
    /// Returns the parsed frame and the number of bytes consumed
    pub fn from_bytes(data: &[u8]) -> Result<(Self, usize)> {
        if data.len() < FRAME_BASE_SIZE {
            return Err(ENFCAPError::CorruptedFrame(0));
        }

        let mut offset = 0;

        // Frame ID (4 bytes)
        let frame_id = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        // Timestamp (4 bytes)
        let timestamp = f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        offset += 4;

        // Ego Transform (48 bytes)
        let ego_transform = EgoTransform::from_bytes(&data[offset..offset + 48]);
        offset += 48;

        // Vehicle State (40 bytes)
        let vehicle_state = VehicleState::from_bytes(&data[offset..offset + 40]);
        offset += 40;

        // Scene Entity Count (2 bytes)
        let entity_count = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap()) as usize;
        offset += 2;

        // Scene Entities (entity_count * 16 bytes)
        let entities_size = entity_count * ENTITY_RECORD_SIZE;
        if data.len() < offset + entities_size {
            return Err(ENFCAPError::CorruptedFrame(offset as u64));
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
            return Err(ENFCAPError::CorruptedFrame(offset as u64));
        }

        let mut road_points = Vec::with_capacity(road_count);
        for i in 0..road_count {
            let road_offset = offset + i * ROAD_POINT_SIZE;
            road_points.push(RoadPoint::from_bytes(&data[road_offset..]));
        }
        offset += roads_size;

        // Frame Flags (2 bytes)
        let flags_bits = u16::from_le_bytes(data[offset..offset + 2].try_into().unwrap());
        let flags = FrameFlags::from_bits_truncate(flags_bits);
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
        bytes.extend_from_slice(&self.flags.bits().to_le_bytes());

        bytes
    }

    /// Calculate the serialized size of this frame
    pub fn serialized_size(&self) -> usize {
        FRAME_BASE_SIZE
            + self.scene_entities.len().min(u16::MAX as usize) * ENTITY_RECORD_SIZE
            + self.road_points.len().min(u16::MAX as usize) * ROAD_POINT_SIZE
    }

    /// Check if this is an anchor (keyframe)
    #[inline]
    pub fn is_anchor(&self) -> bool {
        self.flags.contains(FrameFlags::ANCHOR)
    }

    /// Check if a collision was detected
    #[inline]
    pub fn has_collision(&self) -> bool {
        self.flags.contains(FrameFlags::COLLISION)
    }

    /// Check if this frame has manual (human) control
    #[inline]
    pub fn is_manual_control(&self) -> bool {
        self.flags.contains(FrameFlags::MANUAL_CONTROL)
    }
}

// ============================================================================
// ENFCAPReader - Memory-Mapped File Reader
// ============================================================================

/// High-performance ENFCAP file reader with memory-mapped access
///
/// Provides O(1) random frame access and parallel frame decoding.
#[pyclass]
pub struct ENFCAPReader {
    /// Memory-mapped file data
    mmap: Mmap,
    /// Parsed file header
    header: ENFCAPHeader,
    /// Index table for random access
    index: ENFCAPIndex,
    /// File path for error reporting
    path: String,
}

impl ENFCAPReader {
    /// Open an ENFCAP file for reading
    ///
    /// Memory-maps the file for zero-copy access.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = File::open(&path)?;
        let metadata = file.metadata()?;

        if metadata.len() < ENFCAP_HEADER_SIZE as u64 {
            return Err(ENFCAPError::FileTooSmall(metadata.len() as usize));
        }

        // Memory-map the file
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| ENFCAPError::Mmap(e.to_string()))?
        };

        // Parse header
        let header: ENFCAPHeader = *bytemuck::from_bytes(&mmap[..ENFCAP_HEADER_SIZE]);
        header.validate()?;

        // Calculate index table location
        let index_size = header.frame_count as usize * ENFCAP_INDEX_ENTRY_SIZE;
        let index_offset = mmap.len() - index_size;

        // Parse index table
        let index = ENFCAPIndex::from_bytes(&mmap[index_offset..], header.frame_count)?;

        Ok(Self {
            mmap,
            header,
            index,
            path: path_str,
        })
    }

    /// Get the file header
    pub fn header(&self) -> &ENFCAPHeader {
        &self.header
    }

    /// Get the number of frames in the file
    pub fn frame_count(&self) -> u32 {
        self.header.frame_count
    }

    /// Get the session start timestamp
    pub fn start_timestamp(&self) -> f64 {
        self.header.start_timestamp()
    }

    /// Get header flags
    pub fn flags(&self) -> HeaderFlags {
        self.header.header_flags()
    }

    /// Get number of frames (alias for frame_count)
    pub fn num_frames(&self) -> usize {
        self.header.frame_count as usize
    }

    /// Get frame dimensions (width, height)
    /// Note: ENFCAP format doesn't store dimensions in header, returns default
    pub fn dimensions(&self) -> (usize, usize) {
        // TODO: Read from first frame or store in header
        (1920, 1080)
    }

    /// Get frames per second
    /// Note: ENFCAP format doesn't store FPS in header, returns default
    pub fn fps(&self) -> f32 {
        30.0
    }

    /// Read a single frame by index (O(1) random access)
    pub fn read_frame(&self, frame_index: u32) -> Result<FrameData> {
        if frame_index >= self.header.frame_count {
            return Err(ENFCAPError::FrameOutOfBounds(
                frame_index,
                self.header.frame_count,
            ));
        }

        let offset = self
            .index
            .get_offset(frame_index)
            .ok_or(ENFCAPError::InvalidIndexTable)? as usize;

        let (frame, _) = FrameData::from_bytes(&self.mmap[offset..])?;
        Ok(frame)
    }

    /// Read multiple frames in parallel using rayon
    pub fn read_frames_parallel(&self, range: Range<u32>) -> Result<Vec<FrameData>> {
        let start = range.start;
        let end = range.end.min(self.header.frame_count);

        if start >= end {
            return Ok(Vec::new());
        }

        let entries = self.index.get_range(start..end);
        let mmap = &self.mmap;

        let frames: Result<Vec<FrameData>> = entries
            .par_iter()
            .map(|entry| {
                let offset = entry.offset as usize;
                FrameData::from_bytes(&mmap[offset..]).map(|(f, _)| f)
            })
            .collect();

        frames
    }

    /// Stream frames sequentially (memory-efficient for large files)
    pub fn iter_frames(&self) -> FrameIterator<'_> {
        FrameIterator {
            reader: self,
            current_index: 0,
        }
    }

    /// Get raw frame bytes without parsing (for custom processing)
    pub fn get_frame_bytes(&self, frame_index: u32) -> Result<&[u8]> {
        if frame_index >= self.header.frame_count {
            return Err(ENFCAPError::FrameOutOfBounds(
                frame_index,
                self.header.frame_count,
            ));
        }

        let offset = self
            .index
            .get_offset(frame_index)
            .ok_or(ENFCAPError::InvalidIndexTable)? as usize;

        // Calculate frame size by looking at next offset or end of data
        let next_offset = if frame_index + 1 < self.header.frame_count {
            self.index.get_offset(frame_index + 1).unwrap() as usize
        } else {
            self.mmap.len() - (self.header.frame_count as usize * ENFCAP_INDEX_ENTRY_SIZE)
        };

        Ok(&self.mmap[offset..next_offset])
    }

    /// Get the underlying memory map for advanced use cases
    pub fn raw_data(&self) -> &[u8] {
        &self.mmap
    }

    /// Validate file integrity by checking all frames
    pub fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport {
            frame_count: self.header.frame_count,
            valid_frames: 0,
            invalid_frames: Vec::new(),
            total_entities: 0,
            total_road_points: 0,
            anchor_count: 0,
        };

        for i in 0..self.header.frame_count {
            match self.read_frame(i) {
                Ok(frame) => {
                    report.valid_frames += 1;
                    report.total_entities += frame.scene_entities.len();
                    report.total_road_points += frame.road_points.len();
                    if frame.is_anchor() {
                        report.anchor_count += 1;
                    }
                }
                Err(e) => {
                    report.invalid_frames.push((i, format!("{}", e)));
                }
            }
        }

        Ok(report)
    }
}

/// Frame iterator for sequential access
pub struct FrameIterator<'a> {
    reader: &'a ENFCAPReader,
    current_index: u32,
}

impl<'a> Iterator for FrameIterator<'a> {
    type Item = Result<FrameData>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.reader.header.frame_count {
            return None;
        }

        let result = self.reader.read_frame(self.current_index);
        self.current_index += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.reader.header.frame_count - self.current_index) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for FrameIterator<'a> {}

/// Validation report for file integrity checks
#[derive(Debug)]
pub struct ValidationReport {
    pub frame_count: u32,
    pub valid_frames: u32,
    pub invalid_frames: Vec<(u32, String)>,
    pub total_entities: usize,
    pub total_road_points: usize,
    pub anchor_count: usize,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.invalid_frames.is_empty()
    }
}

// ============================================================================
// ENFCAPWriter - File Writer with Buffered Output
// ============================================================================

/// ENFCAP file writer with buffered output and automatic finalization
pub struct ENFCAPWriter {
    /// Buffered file writer
    writer: BufWriter<File>,
    /// File path
    path: String,
    /// Frame offsets for index table
    index: ENFCAPIndex,
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

impl ENFCAPWriter {
    /// Create a new ENFCAP file for writing
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::create_with_options(path, HeaderFlags::ANCHOR_FRAMES, 100)
    }

    /// Create a new ENFCAP file with custom options
    pub fn create_with_options<P: AsRef<Path>>(
        path: P,
        flags: HeaderFlags,
        anchor_interval: u32,
    ) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy().to_string();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        let mut writer = BufWriter::with_capacity(64 * 1024, file);

        // Write placeholder header
        let header = ENFCAPHeader::new(0, 0.0, flags);
        writer.write_all(bytemuck::bytes_of(&header))?;

        Ok(Self {
            writer,
            path: path_str,
            index: ENFCAPIndex::with_capacity(10000),
            start_timestamp: 0.0,
            flags,
            position: ENFCAP_HEADER_SIZE as u64,
            anchor_interval,
            finalized: false,
        })
    }

    /// Set the session start timestamp
    pub fn set_start_timestamp(&mut self, timestamp: f64) {
        self.start_timestamp = timestamp;
    }

    /// Write a frame to the file
    pub fn write_frame(&mut self, frame: &FrameData) -> Result<()> {
        // Record offset for this frame
        self.index.push(self.position);

        // Check if this should be an anchor frame
        let frame_count = self.index.len() as u32;
        let mut frame = frame.clone();
        if frame_count % self.anchor_interval == 0 {
            frame.flags |= FrameFlags::ANCHOR;
        }

        // Serialize and write frame
        let bytes = frame.to_bytes();
        self.writer.write_all(&bytes)?;
        self.position += bytes.len() as u64;

        Ok(())
    }

    /// Write multiple frames
    pub fn write_frames(&mut self, frames: &[FrameData]) -> Result<()> {
        for frame in frames {
            self.write_frame(frame)?;
        }
        Ok(())
    }

    /// Get the current frame count
    pub fn frame_count(&self) -> u32 {
        self.index.len() as u32
    }

    /// Finalize the file (write index table and update header)
    pub fn finalize(mut self) -> Result<()> {
        self.finalize_internal()?;
        self.finalized = true;
        Ok(())
    }

    fn finalize_internal(&mut self) -> Result<()> {
        // Write index table
        let index_bytes = self.index.to_bytes();
        self.writer.write_all(&index_bytes)?;

        // Flush all buffered data
        self.writer.flush()?;

        // Update header with final values
        let frame_count = self.frame_count();
        let start_timestamp = self.start_timestamp;
        let flags = self.flags;

        let file = self.writer.get_mut();
        file.seek(SeekFrom::Start(0))?;

        let header = ENFCAPHeader::new(frame_count, start_timestamp, flags);
        file.write_all(bytemuck::bytes_of(&header))?;

        file.flush()?;

        Ok(())
    }

    /// Get the file path
    pub fn path(&self) -> &str {
        &self.path
    }
}

impl Drop for ENFCAPWriter {
    fn drop(&mut self) {
        if !self.finalized && self.index.len() > 0 {
            // Attempt to finalize on drop, but ignore errors
            let _ = self.finalize_internal();
        }
    }
}

// ============================================================================
// Streaming Reader for Memory-Constrained Environments
// ============================================================================

/// Streaming reader for large files that don't fit in memory
pub struct ENFCAPStreamReader {
    file: File,
    header: ENFCAPHeader,
    index: ENFCAPIndex,
    buffer: Vec<u8>,
}

impl ENFCAPStreamReader {
    /// Open a file for streaming access
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mut file = File::open(&path)?;
        let metadata = file.metadata()?;

        if metadata.len() < ENFCAP_HEADER_SIZE as u64 {
            return Err(ENFCAPError::FileTooSmall(metadata.len() as usize));
        }

        // Read header
        let mut header_bytes = [0u8; ENFCAP_HEADER_SIZE];
        file.read_exact(&mut header_bytes)?;
        let header: ENFCAPHeader = *bytemuck::from_bytes(&header_bytes);
        header.validate()?;

        // Read index table
        let index_size = header.frame_count as usize * ENFCAP_INDEX_ENTRY_SIZE;
        let index_offset = metadata.len() as usize - index_size;

        file.seek(SeekFrom::Start(index_offset as u64))?;
        let mut index_bytes = vec![0u8; index_size];
        file.read_exact(&mut index_bytes)?;

        let index = ENFCAPIndex::from_bytes(&index_bytes, header.frame_count)?;

        Ok(Self {
            file,
            header,
            index,
            buffer: Vec::with_capacity(4096),
        })
    }

    /// Read a single frame (seeks to position and reads)
    pub fn read_frame(&mut self, frame_index: u32) -> Result<FrameData> {
        if frame_index >= self.header.frame_count {
            return Err(ENFCAPError::FrameOutOfBounds(
                frame_index,
                self.header.frame_count,
            ));
        }

        let offset = self
            .index
            .get_offset(frame_index)
            .ok_or(ENFCAPError::InvalidIndexTable)?;

        // Calculate frame size
        let next_offset = if frame_index + 1 < self.header.frame_count {
            self.index.get_offset(frame_index + 1).unwrap()
        } else {
            (self.file.metadata()?.len() as u64)
                - (self.header.frame_count as u64 * ENFCAP_INDEX_ENTRY_SIZE as u64)
        };

        let frame_size = (next_offset - offset) as usize;

        // Resize buffer if needed
        if self.buffer.len() < frame_size {
            self.buffer.resize(frame_size, 0);
        }

        // Seek and read
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut self.buffer[..frame_size])?;

        // Parse frame
        let (frame, _) = FrameData::from_bytes(&self.buffer[..frame_size])?;
        Ok(frame)
    }

    /// Get frame count
    pub fn frame_count(&self) -> u32 {
        self.header.frame_count
    }

    /// Get header
    pub fn header(&self) -> &ENFCAPHeader {
        &self.header
    }
}

// ============================================================================
// Parallel Frame Decoder
// ============================================================================

/// Parallel frame decoder for batch processing
pub struct ParallelDecoder {
    thread_pool: rayon::ThreadPool,
}

impl ParallelDecoder {
    /// Create a new parallel decoder with the specified number of threads
    pub fn new(num_threads: usize) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .expect("Failed to create thread pool");

        Self { thread_pool }
    }

    /// Decode frames in parallel from raw bytes
    pub fn decode_frames(&self, data: &[u8], offsets: &[u64]) -> Vec<Result<FrameData>> {
        self.thread_pool.install(|| {
            offsets
                .par_iter()
                .map(|&offset| {
                    let offset = offset as usize;
                    FrameData::from_bytes(&data[offset..]).map(|(f, _)| f)
                })
                .collect()
        })
    }

    /// Process frames with a custom function in parallel
    pub fn process_frames<F, T>(&self, reader: &ENFCAPReader, range: Range<u32>, f: F) -> Vec<T>
    where
        F: Fn(&FrameData) -> T + Sync,
        T: Send,
    {
        let start = range.start;
        let end = range.end.min(reader.frame_count());

        if start >= end {
            return Vec::new();
        }

        self.thread_pool.install(|| {
            (start..end)
                .into_par_iter()
                .filter_map(|i| reader.read_frame(i).ok())
                .map(|frame| f(&frame))
                .collect()
        })
    }
}

impl Default for ParallelDecoder {
    fn default() -> Self {
        Self::new(rayon::current_num_threads())
    }
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
        assert_eq!(std::mem::size_of::<ENFCAPHeader>(), ENFCAP_HEADER_SIZE);
    }

    #[test]
    fn test_header_roundtrip() {
        let header = ENFCAPHeader::new(100, 12345.678, HeaderFlags::COMPRESSED);
        assert_eq!(header.magic, ENFCAP_MAGIC);
        assert_eq!(header.version, ENFCAP_VERSION);
        assert_eq!(header.frame_count, 100);
        assert!(header.header_flags().contains(HeaderFlags::COMPRESSED));
    }

    #[test]
    fn test_frame_roundtrip() {
        let mut frame = FrameData {
            frame_id: 42,
            timestamp: 1234.5,
            ego_transform: EgoTransform {
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
                entity_type: EntityType::Vehicle,
                position: [100.0, 0.0, 50.0],
                yaw: 1.57,
                velocity: 30.0,
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
        let (parsed, size) = FrameData::from_bytes(&bytes).unwrap();

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
            let mut writer = ENFCAPWriter::create(&path).unwrap();
            writer.set_start_timestamp(1000.0);

            for i in 0..10 {
                let frame = FrameData {
                    frame_id: i,
                    timestamp: i as f32 * 100.0,
                    ..Default::default()
                };
                writer.write_frame(&frame).unwrap();
            }

            writer.finalize().unwrap();
        }

        // Read them back
        {
            let reader = ENFCAPReader::open(&path).unwrap();
            assert_eq!(reader.frame_count(), 10);

            for i in 0..10 {
                let frame = reader.read_frame(i).unwrap();
                assert_eq!(frame.frame_id, i);
                assert_eq!(frame.timestamp, i as f32 * 100.0);
            }
        }
    }

    #[test]
    fn test_parallel_read() {
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_path_buf();

        // Write frames
        {
            let mut writer = ENFCAPWriter::create(&path).unwrap();

            for i in 0..100 {
                let frame = FrameData {
                    frame_id: i,
                    timestamp: i as f32 * 10.0,
                    ..Default::default()
                };
                writer.write_frame(&frame).unwrap();
            }

            writer.finalize().unwrap();
        }

        // Read in parallel
        {
            let reader = ENFCAPReader::open(&path).unwrap();
            let frames = reader.read_frames_parallel(0..100).unwrap();

            assert_eq!(frames.len(), 100);
            for (i, frame) in frames.iter().enumerate() {
                assert_eq!(frame.frame_id, i as u32);
            }
        }
    }

    #[test]
    fn test_index_operations() {
        let mut index = ENFCAPIndex::with_capacity(10);

        for i in 0..10 {
            index.push(i as u64 * 1000);
        }

        assert_eq!(index.len(), 10);
        assert_eq!(index.get_offset(5), Some(5000));
        assert_eq!(index.get_offset(100), None);

        let bytes = index.to_bytes();
        let parsed = ENFCAPIndex::from_bytes(&bytes, 10).unwrap();
        assert_eq!(parsed.len(), 10);
        assert_eq!(parsed.get_offset(5), Some(5000));
    }
}
