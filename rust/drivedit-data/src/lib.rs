//! DriveDiT High-Performance Data Loading Library
//!
//! A Rust library with Python bindings for efficient data loading in the DriveDiT
//! autonomous driving world model. Provides 10x+ speedup over pure Python through:
//!
//! - Memory-mapped file access (zero-copy I/O)
//! - SIMD-accelerated JSON parsing
//! - Parallel image loading with rayon
//! - Efficient caching and prefetching
//! - Zero-copy numpy array creation
//!
//! # Python Usage
//!
//! ```python
//! from drivedit_data import (
//!     EnfusionSession,
//!     EnfusionDataset,
//!     TelemetryParser,
//!     FrameLoader,
//!     DepthLoader,
//!     ENFCAPReader,
//!     DatasetConfig,
//!     TelemetryConfig,
//!     LoaderConfig,
//! )
//!
//! # Load a single session
//! session = EnfusionSession("/path/to/session")
//! frames = session.get_frames(0)  # [T, C, H, W] numpy array
//! controls = session.get_controls(0)  # [T, 4] numpy array
//!
//! # Load an entire dataset
//! dataset = EnfusionDataset("/path/to/data")
//! for frames, controls, depth in dataset:
//!     # Training loop
//!     pass
//!
//! # Direct frame loading
//! loader = FrameLoader()
//! frames = loader.load_frames(["frame1.jpg", "frame2.jpg"])
//!
//! # Telemetry parsing
//! parser = TelemetryParser()
//! controls = parser.parse_csv("/path/to/telemetry.csv")
//! ```
//!
//! # Performance Notes
//!
//! - Use `preload_sessions()` to warm up the cache before training
//! - Set `use_mmap=True` in LoaderConfig for large datasets
//! - Use `cache_decoded=True` for repeated access to same frames
//! - Call `set_epoch()` on dataset to reshuffle between epochs

mod config;
mod dataset;
mod enfcap;
mod error;
mod frames;
mod session;
mod telemetry;

use pyo3::prelude::*;

// Re-export main types
pub use config::{DatasetConfig, EnfcapConfig, LoaderConfig, TelemetryConfig};
pub use dataset::{DataLoaderIterator, EnfusionDataset};
pub use enfcap::ENFCAPReader;
pub use error::{DataError, DataResult};
pub use frames::{DepthLoader, FrameLoader};
pub use session::{EnfusionSession, SessionMetadata};
pub use telemetry::TelemetryParser;

/// Python module initialization.
#[pymodule]
fn drivedit_data(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Configuration classes
    m.add_class::<DatasetConfig>()?;
    m.add_class::<TelemetryConfig>()?;
    m.add_class::<LoaderConfig>()?;
    m.add_class::<EnfcapConfig>()?;

    // Main data classes
    m.add_class::<EnfusionSession>()?;
    m.add_class::<EnfusionDataset>()?;
    m.add_class::<SessionMetadata>()?;

    // Loaders and parsers
    m.add_class::<TelemetryParser>()?;
    m.add_class::<FrameLoader>()?;
    m.add_class::<DepthLoader>()?;
    m.add_class::<ENFCAPReader>()?;

    // Iterator
    m.add_class::<DataLoaderIterator>()?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "DriveDiT Team")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = DatasetConfig::default();
        assert_eq!(config.sequence_length, 16);
        assert_eq!(config.control_dim, 4);
    }

    #[test]
    fn test_loader_config_auto_detect() {
        let config = LoaderConfig::auto_detect();
        assert!(config.num_workers > 0);
    }

    #[test]
    fn test_telemetry_config_presets() {
        let comma = TelemetryConfig::comma_ai();
        assert_eq!(comma.sampling_rate_hz, 20.0);

        let nuscenes = TelemetryConfig::nuscenes();
        assert_eq!(nuscenes.sampling_rate_hz, 12.0);
    }
}
