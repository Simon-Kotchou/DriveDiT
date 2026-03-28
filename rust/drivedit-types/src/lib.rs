//! Shared types for the DriveDiT data pipeline.
//!
//! This crate provides unified type definitions used across all DriveDiT crates:
//! - `drivedit-io`: Low-level I/O operations
//! - `drivedit-batch`: Batch loading for training
//! - `drivedit-streaming`: Real-time frame streaming
//! - `drivedit-py`: Python bindings
//!
//! # Features
//!
//! - `serde`: Enable serialization/deserialization support via serde
//!
//! # Example
//!
//! ```rust
//! use drivedit_types::{Telemetry, RawFrame, ProcessedFrame, DriveditError};
//!
//! // Create telemetry data
//! let telemetry = Telemetry {
//!     speed_kmh: 60.0,
//!     steering: 0.1,
//!     ..Default::default()
//! };
//!
//! // Convert to array for numpy
//! let array = telemetry.to_array();
//!
//! // Get 6D control tensor
//! let controls = telemetry.to_control_array();
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod config;
mod error;
mod frame;
mod telemetry;

pub use config::{BatchConfig, NormalizationParams, StreamConfig};
pub use error::{DriveditError, Result};
pub use frame::{FrameBatch, FrameMetadata, ProcessedFrame, RawFrame};
pub use telemetry::{ControlSignal, Telemetry, CONTROL_ARRAY_SIZE, TELEMETRY_ARRAY_SIZE};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default ring buffer capacity (256 frames = 51.2s at 5Hz).
pub const DEFAULT_BUFFER_CAPACITY: usize = 256;

/// Default image width from Enfusion capture.
pub const DEFAULT_WIDTH: u32 = 640;

/// Default image height from Enfusion capture.
pub const DEFAULT_HEIGHT: u32 = 360;

/// Default frame rate for capture (Hz).
pub const DEFAULT_FPS: f32 = 5.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_BUFFER_CAPACITY, 256);
        assert_eq!(DEFAULT_WIDTH, 640);
        assert_eq!(DEFAULT_HEIGHT, 360);
        assert!((DEFAULT_FPS - 5.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_raw_frame_creation() {
        let telemetry = Telemetry::default();
        let frame = RawFrame::new(1, 0.0, vec![0u8; 640 * 360 * 3], 640, 360, telemetry);
        assert!(frame.is_valid());
        assert_eq!(frame.frame_id, 1);
    }

    #[test]
    fn test_processed_frame_creation() {
        let telemetry = Telemetry::default();
        let frame = ProcessedFrame::new(1, 0.0, vec![0.0f32; 3 * 640 * 360], 640, 360, telemetry);
        assert!(frame.is_valid());
        assert_eq!(frame.shape(), (3, 360, 640));
    }

    #[test]
    fn test_telemetry_array_roundtrip() {
        let original = Telemetry {
            pos: [1.0, 2.0, 3.0],
            speed_kmh: 60.0,
            steering: 0.5,
            throttle: 0.8,
            brake: 0.1,
            ..Default::default()
        };
        let arr = original.to_array();
        let restored = Telemetry::from_array(&arr).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_control_array() {
        let telemetry = Telemetry {
            speed_kmh: 72.0, // 20 m/s
            steering: 0.3,
            acceleration: 1.5,
            heading_rate: 0.2,
            ..Default::default()
        };

        let controls = telemetry.to_control_array();
        assert_eq!(controls.len(), CONTROL_ARRAY_SIZE);
        assert_eq!(controls[0], 0.3); // steering
    }

    #[test]
    fn test_stream_config() {
        let config = StreamConfig::default();
        assert_eq!(config.frame_width, DEFAULT_WIDTH);
        assert_eq!(config.frame_height, DEFAULT_HEIGHT);
    }

    #[test]
    fn test_batch_config() {
        let config = BatchConfig::default();
        assert!(config.num_workers > 0);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_normalization_params() {
        let params = NormalizationParams::default();
        assert!((params.mean[0] - 0.485).abs() < 1e-6);
        
        let unit = NormalizationParams::unit_range();
        assert_eq!(unit.mean, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_frame_batch() {
        let mut batch = FrameBatch::with_capacity(4, 64, 64);
        assert!(batch.is_empty());
        
        let frame = RawFrame::new(0, 0.0, vec![0u8; 64*64*3], 64, 64, Telemetry::default());
        batch.push(frame);
        
        assert_eq!(batch.len(), 1);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_error_types() {
        let err = DriveditError::frame("test error");
        assert!(err.to_string().contains("test error"));
        
        let err = DriveditError::index_out_of_bounds(10, 5);
        assert!(err.to_string().contains("10"));
    }
}
