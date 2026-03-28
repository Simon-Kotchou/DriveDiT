//! Real-time streaming data pipeline for DriveDiT.
//!
//! This crate provides async file watching, lock-free ring buffers, and
//! frame-telemetry synchronization for real-time autonomous driving data
//! streaming.
//!
//! # Modules
//!
//! - [`watcher`]: Async file watcher for detecting new frames
//! - [`buffer`]: Lock-free ring buffer for frame storage
//! - [`sync`]: Frame-telemetry synchronization
//! - [`pipeline`]: High-level streaming pipeline
//!
//! # Example
//!
//! ```rust,ignore
//! use drivedit_streaming::{StreamingPipeline, StreamConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = StreamConfig::with_path("/path/to/frames");
//!     let mut pipeline = StreamingPipeline::new(config).await?;
//!
//!     while let Some(frame) = pipeline.next_frame().await? {
//!         println!("Got frame {}", frame.metadata.frame_id);
//!     }
//!     Ok(())
//! }
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod buffer;
pub mod pipeline;
pub mod sync;
pub mod watcher;

// Re-export main types for convenience
pub use buffer::{FrameConsumer, FrameProducer, FrameRingBuffer};
pub use pipeline::StreamingPipeline;
pub use sync::FrameTelemetrySync;
pub use watcher::{FrameEvent, FrameWatcher};

// Re-export types from dependencies
pub use drivedit_io::{decode_bmp, parse_frame_id};
pub use drivedit_types::{
    DriveditError, FrameMetadata, RawFrame, Result, StreamConfig, Telemetry,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reexports() {
        let _config = StreamConfig::default();
        let _meta = FrameMetadata::default();
    }
}
