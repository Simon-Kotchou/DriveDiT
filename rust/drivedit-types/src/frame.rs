//! Unified frame types for the DriveDiT data pipeline.
//!
//! This module provides frame representations at different stages of processing:
//!
//! - [`RawFrame`]: Raw u8 HWC format, suitable for streaming and before normalization
//! - [`ProcessedFrame`]: Normalized f32 CHW format, ready for model training
//! - [`FrameMetadata`]: Lightweight frame information without pixel data
//! - [`FrameBatch`]: Batch of frames for efficient training

use crate::telemetry::{Telemetry, TELEMETRY_ARRAY_SIZE};
use std::path::PathBuf;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Raw video frame with u8 pixel data in HWC format.
///
/// This represents a frame before normalization, suitable for streaming
/// pipelines and disk storage. Pixels are stored in row-major order as
/// `[H, W, C]` where C=3 for RGB.
///
/// # Memory Layout
///
/// Pixels are contiguous bytes in HWC (Height-Width-Channel) order:
/// - Row 0: [R, G, B, R, G, B, ...] for width pixels
/// - Row 1: [R, G, B, R, G, B, ...]
/// - ...
///
/// # Example
///
/// ```rust
/// use drivedit_types::{RawFrame, Telemetry};
///
/// let frame = RawFrame::new(
///     42,                        // frame_id
///     1234.5,                    // timestamp_ms
///     vec![0u8; 640 * 360 * 3],  // RGB pixels
///     640,                       // width
///     360,                       // height
///     Telemetry::default(),
/// );
///
/// assert!(frame.is_valid());
/// assert_eq!(frame.shape(), (360, 640, 3));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RawFrame {
    /// Frame ID (monotonic counter from capture).
    pub frame_id: u64,

    /// Capture timestamp in milliseconds from session start.
    pub timestamp_ms: f64,

    /// RGB pixel data in HWC format [H, W, 3].
    pub pixels: Vec<u8>,

    /// Image width in pixels.
    pub width: u32,

    /// Image height in pixels.
    pub height: u32,

    /// Synchronized vehicle telemetry.
    pub telemetry: Telemetry,
}

impl RawFrame {
    /// Create a new raw frame.
    ///
    /// # Arguments
    ///
    /// * `frame_id` - Monotonic frame counter
    /// * `timestamp_ms` - Capture timestamp in milliseconds
    /// * `pixels` - RGB pixel data in row-major HWC layout
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `telemetry` - Synchronized vehicle telemetry
    pub fn new(
        frame_id: u64,
        timestamp_ms: f64,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        telemetry: Telemetry,
    ) -> Self {
        Self {
            frame_id,
            timestamp_ms,
            pixels,
            width,
            height,
            telemetry,
        }
    }

    /// Create an empty frame with specified dimensions.
    pub fn empty(width: u32, height: u32) -> Self {
        let size = (width as usize) * (height as usize) * 3;
        Self {
            frame_id: 0,
            timestamp_ms: 0.0,
            pixels: vec![0u8; size],
            width,
            height,
            telemetry: Telemetry::default(),
        }
    }

    /// Get the expected pixel buffer size.
    #[inline]
    pub fn expected_size(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 3
    }

    /// Validate frame data integrity.
    ///
    /// Returns `true` if the pixel buffer size matches the expected dimensions.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.pixels.len() == self.expected_size()
    }

    /// Get frame dimensions as (height, width, channels).
    #[inline]
    pub fn shape(&self) -> (u32, u32, u32) {
        (self.height, self.width, 3)
    }

    /// Get a slice view of the pixels.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.pixels
    }

    /// Get a mutable slice view of the pixels.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        &mut self.pixels
    }

    /// Get total pixel count (width * height).
    #[inline]
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Extract metadata from this frame (without pixels).
    pub fn metadata(&self) -> FrameMetadata {
        FrameMetadata {
            frame_id: self.frame_id,
            timestamp_ms: self.timestamp_ms,
            width: self.width,
            height: self.height,
            source_path: None,
        }
    }
}

/// Processed frame with f32 pixel data in CHW format.
///
/// This represents a frame after normalization, ready for model training.
/// Pixels are stored in channel-first order as `[C, H, W]` which is the
/// standard format for PyTorch tensors.
///
/// # Memory Layout
///
/// Pixels are contiguous f32 values in CHW (Channel-Height-Width) order:
/// - Channel 0 (R): All H*W red values
/// - Channel 1 (G): All H*W green values
/// - Channel 2 (B): All H*W blue values
///
/// # Example
///
/// ```rust
/// use drivedit_types::{ProcessedFrame, Telemetry};
///
/// let frame = ProcessedFrame::new(
///     42,                          // frame_id
///     1234.5,                      // timestamp_ms
///     vec![0.0f32; 3 * 360 * 640], // Normalized CHW pixels
///     640,                         // width
///     360,                         // height
///     Telemetry::default(),
/// );
///
/// assert!(frame.is_valid());
/// assert_eq!(frame.shape(), (3, 360, 640));
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ProcessedFrame {
    /// Frame ID (monotonic counter from capture).
    pub frame_id: u64,

    /// Capture timestamp in milliseconds from session start.
    pub timestamp_ms: f64,

    /// Normalized pixel data in CHW format [C, H, W].
    pub pixels: Vec<f32>,

    /// Image width in pixels.
    pub width: u32,

    /// Image height in pixels.
    pub height: u32,

    /// Synchronized vehicle telemetry.
    pub telemetry: Telemetry,
}

impl ProcessedFrame {
    /// Create a new processed frame.
    ///
    /// # Arguments
    ///
    /// * `frame_id` - Monotonic frame counter
    /// * `timestamp_ms` - Capture timestamp in milliseconds
    /// * `pixels` - Normalized f32 pixel data in CHW layout
    /// * `width` - Image width in pixels
    /// * `height` - Image height in pixels
    /// * `telemetry` - Synchronized vehicle telemetry
    pub fn new(
        frame_id: u64,
        timestamp_ms: f64,
        pixels: Vec<f32>,
        width: u32,
        height: u32,
        telemetry: Telemetry,
    ) -> Self {
        Self {
            frame_id,
            timestamp_ms,
            pixels,
            width,
            height,
            telemetry,
        }
    }

    /// Create an empty frame with specified dimensions.
    pub fn empty(width: u32, height: u32) -> Self {
        let size = 3 * (width as usize) * (height as usize);
        Self {
            frame_id: 0,
            timestamp_ms: 0.0,
            pixels: vec![0.0f32; size],
            width,
            height,
            telemetry: Telemetry::default(),
        }
    }

    /// Get the expected pixel buffer size.
    #[inline]
    pub fn expected_size(&self) -> usize {
        3 * (self.width as usize) * (self.height as usize)
    }

    /// Validate frame data integrity.
    ///
    /// Returns `true` if the pixel buffer size matches the expected dimensions.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.pixels.len() == self.expected_size()
    }

    /// Get frame dimensions as (channels, height, width).
    #[inline]
    pub fn shape(&self) -> (u32, u32, u32) {
        (3, self.height, self.width)
    }

    /// Get a slice view of the pixels.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.pixels
    }

    /// Get a mutable slice view of the pixels.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.pixels
    }

    /// Get a raw pointer to the data for FFI.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.pixels.as_ptr()
    }

    /// Get total pixel count (width * height).
    #[inline]
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Consume and return the underlying vector.
    pub fn into_vec(self) -> Vec<f32> {
        self.pixels
    }

    /// Extract metadata from this frame (without pixels).
    pub fn metadata(&self) -> FrameMetadata {
        FrameMetadata {
            frame_id: self.frame_id,
            timestamp_ms: self.timestamp_ms,
            width: self.width,
            height: self.height,
            source_path: None,
        }
    }
}

/// Lightweight frame metadata without pixel data.
///
/// This structure is useful for frame indexing, synchronization, and
/// metadata operations without carrying the heavy pixel payload.
///
/// # Example
///
/// ```rust
/// use drivedit_types::FrameMetadata;
///
/// let meta = FrameMetadata::new(42, 1234.5, 640, 360);
/// assert_eq!(meta.pixel_size(), 640 * 360 * 3);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FrameMetadata {
    /// Frame ID from telemetry (monotonic counter).
    pub frame_id: u64,

    /// Capture timestamp in milliseconds from session start.
    pub timestamp_ms: f64,

    /// Image width in pixels.
    pub width: u32,

    /// Image height in pixels.
    pub height: u32,

    /// Source file path (if loaded from disk).
    pub source_path: Option<PathBuf>,
}

impl FrameMetadata {
    /// Create new frame metadata.
    pub fn new(frame_id: u64, timestamp_ms: f64, width: u32, height: u32) -> Self {
        Self {
            frame_id,
            timestamp_ms,
            width,
            height,
            source_path: None,
        }
    }

    /// Create metadata with source path.
    pub fn with_source(mut self, path: PathBuf) -> Self {
        self.source_path = Some(path);
        self
    }

    /// Get expected pixel buffer size for RGB u8 format.
    #[inline]
    pub fn pixel_size(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 3
    }

    /// Get expected pixel buffer size for RGB f32 format.
    #[inline]
    pub fn pixel_size_f32(&self) -> usize {
        self.pixel_size()
    }

    /// Get dimensions as (height, width).
    #[inline]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.height, self.width)
    }
}

impl Default for FrameMetadata {
    fn default() -> Self {
        Self {
            frame_id: 0,
            timestamp_ms: 0.0,
            width: 0,
            height: 0,
            source_path: None,
        }
    }
}

/// Batch of frames for efficient training.
///
/// Contains contiguous pixel data and telemetry for multiple frames,
/// optimized for direct conversion to numpy arrays.
///
/// # Memory Layout
///
/// - `pixels`: Contiguous bytes in BHWC format `[B, H, W, 3]`
/// - `telemetry`: Flattened telemetry `[B * TELEMETRY_ARRAY_SIZE]`
///
/// # Example
///
/// ```rust
/// use drivedit_types::{FrameBatch, RawFrame, Telemetry};
///
/// let mut batch = FrameBatch::with_capacity(4, 640, 360);
/// assert!(batch.is_empty());
///
/// // Add frames to batch
/// let frame = RawFrame::new(0, 0.0, vec![0u8; 640*360*3], 640, 360, Telemetry::default());
/// batch.push(frame);
///
/// assert_eq!(batch.len(), 1);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FrameBatch {
    /// Contiguous pixel buffer in BHWC format `[B, H, W, 3]`.
    pub pixels: Vec<u8>,

    /// Flattened telemetry data `[B * TELEMETRY_ARRAY_SIZE]`.
    pub telemetry: Vec<f32>,

    /// Frame IDs for each frame in the batch.
    pub frame_ids: Vec<u64>,

    /// Timestamps in milliseconds for each frame.
    pub timestamps: Vec<f64>,

    /// Current batch size (number of frames).
    pub batch_size: usize,

    /// Frame width in pixels.
    pub width: u32,

    /// Frame height in pixels.
    pub height: u32,
}

impl FrameBatch {
    /// Create an empty batch with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of frames
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    pub fn with_capacity(capacity: usize, width: u32, height: u32) -> Self {
        let frame_size = (width as usize) * (height as usize) * 3;
        Self {
            pixels: Vec::with_capacity(capacity * frame_size),
            telemetry: Vec::with_capacity(capacity * TELEMETRY_ARRAY_SIZE),
            frame_ids: Vec::with_capacity(capacity),
            timestamps: Vec::with_capacity(capacity),
            batch_size: 0,
            width,
            height,
        }
    }

    /// Create a new empty batch.
    pub fn new(width: u32, height: u32) -> Self {
        Self::with_capacity(0, width, height)
    }

    /// Add a raw frame to the batch.
    pub fn push(&mut self, frame: RawFrame) {
        debug_assert!(
            frame.width == self.width && frame.height == self.height,
            "Frame dimensions must match batch dimensions"
        );
        self.pixels.extend_from_slice(&frame.pixels);
        self.telemetry.extend_from_slice(&frame.telemetry.to_array());
        self.frame_ids.push(frame.frame_id);
        self.timestamps.push(frame.timestamp_ms);
        self.batch_size += 1;
    }

    /// Get the number of frames in the batch.
    #[inline]
    pub fn len(&self) -> usize {
        self.batch_size
    }

    /// Check if batch is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    /// Clear the batch for reuse.
    pub fn clear(&mut self) {
        self.pixels.clear();
        self.telemetry.clear();
        self.frame_ids.clear();
        self.timestamps.clear();
        self.batch_size = 0;
    }

    /// Get the shape as (batch_size, height, width, channels).
    #[inline]
    pub fn shape(&self) -> (usize, u32, u32, u32) {
        (self.batch_size, self.height, self.width, 3)
    }

    /// Get pixel data as a slice.
    #[inline]
    pub fn pixels_slice(&self) -> &[u8] {
        &self.pixels
    }

    /// Get telemetry data as a slice.
    #[inline]
    pub fn telemetry_slice(&self) -> &[f32] {
        &self.telemetry
    }

    /// Get expected size per frame in bytes.
    #[inline]
    pub fn frame_size(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 3
    }
}

impl Default for FrameBatch {
    fn default() -> Self {
        Self::new(640, 360)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_raw_frame_creation() {
        let frame = RawFrame::new(
            42,
            1234.5,
            vec![128u8; 640 * 360 * 3],
            640,
            360,
            Telemetry::default(),
        );

        assert_eq!(frame.frame_id, 42);
        assert!((frame.timestamp_ms - 1234.5).abs() < f64::EPSILON);
        assert!(frame.is_valid());
        assert_eq!(frame.shape(), (360, 640, 3));
        assert_eq!(frame.pixel_count(), 640 * 360);
    }

    #[test]
    fn test_raw_frame_invalid() {
        let frame = RawFrame::new(
            0,
            0.0,
            vec![0u8; 100], // Wrong size
            640,
            360,
            Telemetry::default(),
        );

        assert!(!frame.is_valid());
    }

    #[test]
    fn test_processed_frame_creation() {
        let frame = ProcessedFrame::new(
            42,
            1234.5,
            vec![0.5f32; 3 * 360 * 640],
            640,
            360,
            Telemetry::default(),
        );

        assert_eq!(frame.frame_id, 42);
        assert!(frame.is_valid());
        assert_eq!(frame.shape(), (3, 360, 640));
    }

    #[test]
    fn test_frame_metadata() {
        let meta = FrameMetadata::new(42, 1234.5, 640, 360);

        assert_eq!(meta.frame_id, 42);
        assert_eq!(meta.pixel_size(), 640 * 360 * 3);
        assert_eq!(meta.dimensions(), (360, 640));
        assert!(meta.source_path.is_none());

        let meta_with_path = meta.with_source(PathBuf::from("/path/to/frame.png"));
        assert!(meta_with_path.source_path.is_some());
    }

    #[test]
    fn test_frame_batch() {
        let mut batch = FrameBatch::with_capacity(4, 64, 64);

        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);

        let frame = RawFrame::new(
            0,
            0.0,
            vec![0u8; 64 * 64 * 3],
            64,
            64,
            Telemetry::default(),
        );

        batch.push(frame);
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
        assert_eq!(batch.shape(), (1, 64, 64, 3));

        batch.clear();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_empty_frames() {
        let raw = RawFrame::empty(100, 100);
        assert!(raw.is_valid());
        assert_eq!(raw.pixels.len(), 100 * 100 * 3);

        let processed = ProcessedFrame::empty(100, 100);
        assert!(processed.is_valid());
        assert_eq!(processed.pixels.len(), 3 * 100 * 100);
    }

    #[test]
    fn test_frame_batch_telemetry() {
        let mut batch = FrameBatch::with_capacity(2, 64, 64);
        
        let telemetry = Telemetry {
            speed_kmh: 60.0,
            steering: 0.5,
            ..Default::default()
        };
        
        let frame = RawFrame::new(
            1,
            100.0,
            vec![0u8; 64 * 64 * 3],
            64,
            64,
            telemetry,
        );

        batch.push(frame);
        
        assert_eq!(batch.telemetry.len(), TELEMETRY_ARRAY_SIZE);
        assert_eq!(batch.telemetry[12], 60.0); // speed_kmh
        assert_eq!(batch.telemetry[13], 0.5);  // steering
    }
}
