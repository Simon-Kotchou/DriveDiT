//! Low-level I/O operations for DriveDiT data pipeline.
//!
//! This crate provides unified image decoding, CSV parsing, and binary format
//! handling for the DriveDiT autonomous driving data pipeline.
//!
//! # Modules
//!
//! - [`bmp`]: BMP image decoding optimized for Enfusion capture format
//! - [`jpeg`]: JPEG and PNG decoding with auto-format detection
//! - [`csv`]: Telemetry CSV parsing with frame-indexed lookup
//! - [`enfcap`]: ENFCAP binary format for high-performance data storage
//! - [`mmap`]: Memory-mapped file utilities
//!
//! # Example
//!
//! ```rust,ignore
//! use drivedit_io::{decode_bmp, decode_auto, TelemetryParser};
//! use std::path::Path;
//!
//! // Decode a BMP image file
//! let frame = decode_bmp(Path::new("frame.bmp"))?;
//! println!("Decoded {}x{} frame", frame.width, frame.height);
//!
//! // Auto-detect and decode any supported image format
//! let frame = decode_auto(&image_bytes)?;
//!
//! // Parse telemetry CSV
//! let mut parser = TelemetryParser::new(Path::new("telemetry.csv"));
//! parser.parse_updates()?;
//! if let Some((ts, telemetry)) = parser.get(42) {
//!     println!("Speed: {} km/h", telemetry.speed_kmh);
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod error;

pub mod bmp;
pub mod csv;
pub mod enfcap;
pub mod jpeg;
pub mod mmap;

// Re-exports for convenience
pub use bmp::{decode_bmp, decode_bmp_bytes, BmpDecoder};
pub use csv::{parse_telemetry_csv, TelemetryParser};
pub use enfcap::{ENFCAPReader, ENFCAPWriter, FrameData, HeaderFlags};
pub use jpeg::{decode_auto, decode_jpeg, decode_png};
pub use mmap::{MappedFile, MappedRegion};

// Re-export error types
pub use error::{IoError, Result};

// Re-export types from drivedit-types
pub use drivedit_types::{DriveditError, RawFrame, Telemetry};

/// Parse frame ID from a filename like "frame_00123.bmp" -> 123
///
/// Returns None if the filename doesn't match the expected pattern.
///
/// # Example
///
/// ```rust
/// use drivedit_io::parse_frame_id;
///
/// assert_eq!(parse_frame_id("frame_00042.bmp"), Some(42));
/// assert_eq!(parse_frame_id("frame_1.bmp"), Some(1));
/// assert_eq!(parse_frame_id("invalid.png"), None);
/// ```
pub fn parse_frame_id(filename: &str) -> Option<u64> {
    // Strip extension
    let name = filename.strip_suffix(".bmp")?;

    // Strip "frame_" prefix
    let num_str = name.strip_prefix("frame_")?;

    // Parse the number (handles leading zeros)
    num_str.parse().ok()
}

/// Check if a filename looks like a frame file
pub fn is_frame_file(filename: &str) -> bool {
    filename.starts_with("frame_") && filename.ends_with(".bmp")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frame_id() {
        assert_eq!(parse_frame_id("frame_00001.bmp"), Some(1));
        assert_eq!(parse_frame_id("frame_00123.bmp"), Some(123));
        assert_eq!(parse_frame_id("frame_1.bmp"), Some(1));
        assert_eq!(parse_frame_id("frame_99999.bmp"), Some(99999));
    }

    #[test]
    fn test_parse_frame_id_invalid() {
        assert_eq!(parse_frame_id("invalid.bmp"), None);
        assert_eq!(parse_frame_id("frame_abc.bmp"), None);
        assert_eq!(parse_frame_id("frame_001.png"), None);
        assert_eq!(parse_frame_id(""), None);
    }

    #[test]
    fn test_is_frame_file() {
        assert!(is_frame_file("frame_00001.bmp"));
        assert!(is_frame_file("frame_1.bmp"));
        assert!(!is_frame_file("other.bmp"));
        assert!(!is_frame_file("frame_1.png"));
    }
}
