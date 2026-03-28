//! Batch data loading for DriveDiT training.
//!
//! This crate provides session management, dataset iteration, and parallel frame
//! loading for training the DriveDiT world model.
//!
//! # Modules
//!
//! - [`session`]: Enfusion session management for training data access
//! - [`dataset`]: High-performance dataset with parallel loading
//! - [`loader`]: Parallel frame loading with caching
//! - [`depth`]: Depth map loading with NPZ/PNG support
//! - [`anchor`]: Anchor frame detection with SIMD acceleration
//! - [`simd`]: SIMD utilities for normalization
//!
//! # Example
//!
//! ```rust,ignore
//! use drivedit_batch::{EnfusionSession, EnfusionDataset, FrameLoader};
//! use drivedit_types::BatchConfig;
//! use std::path::Path;
//!
//! // Load a single session
//! let session = EnfusionSession::from_path(
//!     Path::new("./data/session_001"),
//!     BatchConfig::default(),
//! )?;
//!
//! println!("Session has {} frames", session.num_frames());
//!
//! // Create a dataset from multiple sessions
//! let dataset = EnfusionDataset::from_path(
//!     Path::new("./data"),
//!     BatchConfig::default(),
//! )?;
//!
//! // Iterate over samples
//! for idx in 0..dataset.num_samples() {
//!     let (frames, controls, depth) = dataset.get_sample(idx)?;
//!     // Process batch...
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod error;

pub mod anchor;
pub mod dataset;
pub mod depth;
pub mod loader;
pub mod session;
pub mod simd;

// Re-exports
pub use anchor::{AnchorConfig, AnchorDetector, ControlWeights};
pub use dataset::EnfusionDataset;
pub use depth::{DepthLoader, DepthLoaderConfig, DepthMap};
pub use error::{BatchError, Result};
pub use loader::FrameLoader;
pub use session::{find_sessions, is_session_dir, EnfusionSession, SessionMetadata, SessionSample};
pub use simd::{normalize_batch, normalize_frame};

// Re-export types from drivedit-types for convenience
pub use drivedit_types::{
    BatchConfig, ControlSignal, FrameBatch, NormalizationParams, ProcessedFrame, RawFrame,
    StreamConfig, Telemetry,
};

// Re-export I/O utilities from drivedit-io
pub use drivedit_io::{
    decode_auto, decode_bmp, decode_jpeg, decode_png, parse_frame_id, ENFCAPReader, TelemetryParser,
};
