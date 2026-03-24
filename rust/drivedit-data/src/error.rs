//! Error types for DriveDiT data loading library.
//!
//! Uses thiserror for ergonomic error handling with automatic
//! conversion to Python exceptions via PyO3.

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::PyErr;
use thiserror::Error;

/// Main error type for all data loading operations.
#[derive(Error, Debug)]
pub enum DataError {
    /// I/O errors from file operations
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// CSV parsing errors
    #[error("CSV parsing error: {0}")]
    Csv(#[from] csv::Error),

    /// JSON parsing errors
    #[error("JSON parsing error: {0}")]
    Json(String),

    /// Image decoding errors
    #[error("Image error: {0}")]
    Image(#[from] image::ImageError),

    /// Configuration validation errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Data format errors
    #[error("Data format error: {0}")]
    Format(String),

    /// Index out of bounds
    #[error("Index out of bounds: {index} >= {length}")]
    IndexOutOfBounds { index: usize, length: usize },

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),

    /// Invalid shape for array operations
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Memory mapping errors
    #[error("Memory mapping error: {0}")]
    Mmap(String),

    /// Thread pool errors
    #[error("Thread pool error: {0}")]
    ThreadPool(String),

    /// Session state errors
    #[error("Session error: {0}")]
    Session(String),

    /// Telemetry parsing errors
    #[error("Telemetry error: {0}")]
    Telemetry(String),

    /// Frame loading errors
    #[error("Frame loading error: {0}")]
    Frame(String),

    /// Depth data errors
    #[error("Depth loading error: {0}")]
    Depth(String),

    /// ENFCAP format errors
    #[error("ENFCAP error: {0}")]
    Enfcap(String),

    /// Compression/decompression errors
    #[error("Compression error: {0}")]
    Compression(String),
}

/// Result type alias for data operations
pub type DataResult<T> = Result<T, DataError>;

// Conversion from simd_json errors
impl From<simd_json::Error> for DataError {
    fn from(err: simd_json::Error) -> Self {
        DataError::Json(err.to_string())
    }
}

// Conversion to Python exceptions
impl From<DataError> for PyErr {
    fn from(err: DataError) -> PyErr {
        match err {
            DataError::Io(_) => PyIOError::new_err(err.to_string()),
            DataError::Config(_) | DataError::Format(_) | DataError::MissingField(_) => {
                PyValueError::new_err(err.to_string())
            }
            DataError::IndexOutOfBounds { .. } | DataError::ShapeMismatch { .. } => {
                PyValueError::new_err(err.to_string())
            }
            _ => PyRuntimeError::new_err(err.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DataError::IndexOutOfBounds {
            index: 10,
            length: 5,
        };
        assert!(err.to_string().contains("10"));
        assert!(err.to_string().contains("5"));
    }

    #[test]
    fn test_shape_mismatch() {
        let err = DataError::ShapeMismatch {
            expected: vec![3, 224, 224],
            actual: vec![3, 256, 256],
        };
        assert!(err.to_string().contains("224"));
        assert!(err.to_string().contains("256"));
    }
}
