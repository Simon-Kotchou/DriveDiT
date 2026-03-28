//! DriveDiT Unified Python Bindings
//!
//! High-performance data loading library with both batch and streaming APIs
//! for autonomous driving world model training.
//!
//! # Python Usage
//!
//! ```python
//! import drivedit
//!
//! # Batch loading from stored datasets
//! dataset = drivedit.EnfusionDataset("/path/to/data")
//! session = drivedit.EnfusionSession("/path/to/session")
//!
//! # Streaming from live capture
//! stream = drivedit.StreamingLoader("/path/to/live")
//!
//! # Or with explicit submodules
//! from drivedit.batch import EnfusionDataset, EnfusionSession
//! from drivedit.streaming import StreamingLoader
//! ```
//!
//! # Module Structure
//!
//! - `drivedit` - Top-level module with convenience re-exports
//! - `drivedit.batch` - Batch loading APIs for stored datasets
//! - `drivedit.streaming` - Real-time streaming APIs for live capture
//! - `drivedit.types` - Shared type wrappers (Telemetry, Frame, etc.)

mod batch;
mod numpy_utils;
mod streaming;
mod types;

use pyo3::prelude::*;
use tracing_subscriber::EnvFilter;

/// Initialize logging from RUST_LOG environment variable.
fn init_logging() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();
}

/// DriveDiT Python module.
///
/// Provides unified access to batch and streaming data loading APIs.
#[pymodule]
fn drivedit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    init_logging();

    // Add version and metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "DriveDiT Team")?;
    m.add("__doc__", "High-performance data pipeline for DriveDiT autonomous driving world model")?;

    // Add constants
    m.add("DEFAULT_BUFFER_SIZE", 256)?;
    m.add("DEFAULT_WIDTH", 640)?;
    m.add("DEFAULT_HEIGHT", 360)?;
    m.add("TELEMETRY_SIZE", 20)?;

    // Register shared types at top level
    m.add_class::<types::PyTelemetry>()?;
    m.add_class::<types::PyFrame>()?;
    m.add_class::<types::PyFrameBatch>()?;

    // Register batch submodule
    let batch_module = PyModule::new(m.py(), "batch")?;
    batch::register_batch_module(&batch_module)?;
    m.add_submodule(&batch_module)?;

    // Register streaming submodule
    let streaming_module = PyModule::new(m.py(), "streaming")?;
    streaming::register_streaming_module(&streaming_module)?;
    m.add_submodule(&streaming_module)?;

    // Register types submodule
    let types_module = PyModule::new(m.py(), "types")?;
    types::register_types_module(&types_module)?;
    m.add_submodule(&types_module)?;

    // Re-export main classes at top level for convenience
    // These allow: from drivedit import EnfusionDataset
    m.add_class::<batch::PyEnfusionSession>()?;
    m.add_class::<batch::PyEnfusionDataset>()?;
    m.add_class::<batch::PyFrameLoader>()?;
    m.add_class::<batch::PyDepthLoader>()?;
    m.add_class::<batch::PyDatasetConfig>()?;
    m.add_class::<batch::PyLoaderConfig>()?;
    m.add_class::<batch::PyTelemetryConfig>()?;

    m.add_class::<streaming::PyStreamingLoader>()?;

    // Utility functions at top level
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_utils::hwc_to_chw, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_utils::chw_to_hwc, m)?)?;
    m.add_function(wrap_pyfunction!(numpy_utils::normalize_frames, m)?)?;

    Ok(())
}

/// Get the library version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(!v.is_empty());
    }
}
