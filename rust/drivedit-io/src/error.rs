//! Error types for drivedit-io operations.

use thiserror::Error;

/// Errors specific to I/O operations.
#[derive(Error, Debug)]
pub enum IoError {
    /// Invalid BMP header.
    #[error("Invalid BMP header: {0}")]
    InvalidBmpHeader(String),

    /// Unsupported BMP format.
    #[error("Unsupported BMP format: {0}")]
    UnsupportedBmpFormat(String),

    /// Buffer too small.
    #[error("Buffer too small: expected {expected}, got {actual}")]
    BufferTooSmall {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    /// Invalid image dimensions.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Width.
        width: i32,
        /// Height.
        height: i32,
    },

    /// File I/O error.
    #[error("File I/O error: {0}")]
    FileIo(#[from] std::io::Error),

    /// Image decoding error.
    #[error("Image decode error: {0}")]
    ImageDecode(String),

    /// Invalid file header.
    #[error("Invalid header: {0}")]
    InvalidHeader(String),

    /// Index out of bounds.
    #[error("Index out of bounds: {index} >= {max}")]
    IndexOutOfBounds {
        /// The requested index.
        index: usize,
        /// Maximum valid index.
        max: usize,
    },

    /// CSV parsing error.
    #[error("CSV parse error: {0}")]
    CsvParse(String),
}

/// Result type for I/O operations.
pub type Result<T> = std::result::Result<T, IoError>;

impl From<IoError> for drivedit_types::DriveditError {
    fn from(err: IoError) -> Self {
        drivedit_types::DriveditError::ImageDecode(err.to_string())
    }
}
