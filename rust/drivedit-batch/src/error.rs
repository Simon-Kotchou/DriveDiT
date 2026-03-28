//! Error types for drivedit-batch operations.

use thiserror::Error;

/// Errors specific to batch loading operations.
#[derive(Error, Debug)]
pub enum BatchError {
    /// Session error.
    #[error("Session error: {0}")]
    Session(String),

    /// Frame loading error.
    #[error("Frame error: {0}")]
    Frame(String),

    /// Depth loading error.
    #[error("Depth error: {0}")]
    Depth(String),

    /// Telemetry error.
    #[error("Telemetry error: {0}")]
    Telemetry(String),

    /// Configuration error.
    #[error("Config error: {0}")]
    Config(String),

    /// Index out of bounds.
    #[error("Index {index} out of bounds (length {length})")]
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// Maximum valid length.
        length: usize,
    },

    /// I/O error from drivedit-io.
    #[error("I/O error: {0}")]
    Io(#[from] drivedit_io::IoError),

    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Insufficient frames.
    #[error("Insufficient frames: need at least {min}, got {actual}")]
    InsufficientFrames {
        /// Minimum required frames.
        min: usize,
        /// Actual number of frames.
        actual: usize,
    },

    /// Anchor detection error.
    #[error("Anchor error: {0}")]
    Anchor(String),
}

/// Result type for batch operations.
pub type Result<T> = std::result::Result<T, BatchError>;

impl From<std::io::Error> for BatchError {
    fn from(err: std::io::Error) -> Self {
        BatchError::Io(drivedit_io::IoError::FileIo(err))
    }
}

impl From<drivedit_types::DriveditError> for BatchError {
    fn from(err: drivedit_types::DriveditError) -> Self {
        match err {
            drivedit_types::DriveditError::Frame(msg) => BatchError::Frame(msg),
            drivedit_types::DriveditError::TelemetrySync(msg) => BatchError::Telemetry(msg),
            drivedit_types::DriveditError::Config(msg) => BatchError::Config(msg),
            drivedit_types::DriveditError::IndexOutOfBounds { index, max } => {
                BatchError::IndexOutOfBounds { index, length: max }
            }
            _ => BatchError::Session(err.to_string()),
        }
    }
}
