//! Unified error types for DriveDiT operations.
//!
//! This module provides a comprehensive error type that covers all error
//! scenarios across the DriveDiT data pipeline, from I/O operations to
//! frame processing and streaming.

use thiserror::Error;

/// Unified error type for all DriveDiT operations.
///
/// This error type covers all possible failures across the data pipeline:
/// - File I/O errors
/// - Image decoding/encoding errors
/// - CSV/telemetry parsing errors
/// - Buffer operations
/// - Streaming/sync errors
///
/// # Example
///
/// ```rust
/// use drivedit_types::{DriveditError, Result};
///
/// fn load_frame(path: &str) -> Result<()> {
///     Err(DriveditError::frame("Frame not found"))
/// }
/// ```
#[derive(Error, Debug)]
pub enum DriveditError {
    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Image decoding error.
    #[error("Image decode error: {0}")]
    ImageDecode(String),

    /// Invalid image format or dimensions.
    #[error("Invalid image format: {0}")]
    InvalidFormat(String),

    /// CSV parsing error.
    #[error("CSV parsing error: {0}")]
    CsvParse(String),

    /// Telemetry synchronization error.
    #[error("Telemetry sync error: {0}")]
    TelemetrySync(String),

    /// Frame index out of bounds.
    #[error("Frame index {index} out of bounds (max: {max})")]
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// The maximum valid index.
        max: usize,
    },

    /// Invalid header or magic bytes in file format.
    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    /// Corrupted data detected at specified offset.
    #[error("Corrupted data at offset {0}")]
    CorruptedData(u64),

    /// Memory mapping error.
    #[error("Memory map error: {0}")]
    Mmap(String),

    /// Configuration validation error.
    #[error("Configuration error: {0}")]
    Config(String),

    /// Buffer overflow or underflow.
    #[error("Buffer error: {0}")]
    Buffer(String),

    /// Operation timed out.
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// Channel communication error.
    #[error("Channel error: {0}")]
    Channel(String),

    /// File system watcher error.
    #[error("Watcher error: {0}")]
    Watcher(String),

    /// Session not found.
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    /// Session state error.
    #[error("Session error: {0}")]
    Session(String),

    /// Frame loading error.
    #[error("Frame loading error: {0}")]
    Frame(String),

    /// Depth data loading error.
    #[error("Depth loading error: {0}")]
    Depth(String),

    /// Anchor frame detection error.
    #[error("Anchor error: {0}")]
    Anchor(String),

    /// Shape mismatch between expected and actual dimensions.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected shape dimensions.
        expected: Vec<usize>,
        /// Actual shape dimensions.
        actual: Vec<usize>,
    },

    /// Missing required field in data.
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Compression or decompression error.
    #[error("Compression error: {0}")]
    Compression(String),

    /// ENFCAP binary format error.
    #[error("ENFCAP format error: {0}")]
    EnfcapFormat(String),

    /// Thread pool execution error.
    #[error("Thread pool error: {0}")]
    ThreadPool(String),

    /// JSON parsing error.
    #[error("JSON error: {0}")]
    Json(String),
}

impl DriveditError {
    /// Create a new configuration error.
    pub fn config(msg: impl Into<String>) -> Self {
        DriveditError::Config(msg.into())
    }

    /// Create a new session error.
    pub fn session(msg: impl Into<String>) -> Self {
        DriveditError::Session(msg.into())
    }

    /// Create a new frame loading error.
    pub fn frame(msg: impl Into<String>) -> Self {
        DriveditError::Frame(msg.into())
    }

    /// Create a new depth loading error.
    pub fn depth(msg: impl Into<String>) -> Self {
        DriveditError::Depth(msg.into())
    }

    /// Create a new buffer error.
    pub fn buffer(msg: impl Into<String>) -> Self {
        DriveditError::Buffer(msg.into())
    }

    /// Create a new timeout error.
    pub fn timeout(msg: impl Into<String>) -> Self {
        DriveditError::Timeout(msg.into())
    }

    /// Create a new watcher error.
    pub fn watcher(msg: impl Into<String>) -> Self {
        DriveditError::Watcher(msg.into())
    }

    /// Create a new CSV parse error.
    pub fn csv(msg: impl Into<String>) -> Self {
        DriveditError::CsvParse(msg.into())
    }

    /// Create a new image decode error.
    pub fn image(msg: impl Into<String>) -> Self {
        DriveditError::ImageDecode(msg.into())
    }

    /// Create an index out of bounds error.
    pub fn index_out_of_bounds(index: usize, max: usize) -> Self {
        DriveditError::IndexOutOfBounds { index, max }
    }

    /// Create a shape mismatch error.
    pub fn shape_mismatch(expected: Vec<usize>, actual: Vec<usize>) -> Self {
        DriveditError::ShapeMismatch { expected, actual }
    }
}

/// Result type alias using DriveditError.
pub type Result<T> = std::result::Result<T, DriveditError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DriveditError::IndexOutOfBounds { index: 10, max: 5 };
        let msg = err.to_string();
        assert!(msg.contains("10"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_shape_mismatch() {
        let err = DriveditError::shape_mismatch(vec![3, 224, 224], vec![3, 256, 256]);
        let msg = err.to_string();
        assert!(msg.contains("224"));
        assert!(msg.contains("256"));
    }

    #[test]
    fn test_config_error() {
        let err = DriveditError::config("invalid parameter");
        assert!(err.to_string().contains("invalid parameter"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: DriveditError = io_err.into();
        assert!(matches!(err, DriveditError::Io(_)));
    }

    #[test]
    fn test_frame_error() {
        let err = DriveditError::frame("frame corrupted");
        assert!(err.to_string().contains("frame corrupted"));
    }

    #[test]
    fn test_timeout_error() {
        let err = DriveditError::timeout("operation exceeded 5s limit");
        assert!(err.to_string().contains("5s"));
    }
}
