//! High-level streaming pipeline.
//!
//! This module provides a complete streaming pipeline that combines
//! file watching, frame decoding, ring buffer storage, and telemetry
//! synchronization.
//!
//! # Example
//!
//! ```rust,ignore
//! use drivedit_streaming::{StreamingPipeline, StreamConfig};
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = StreamConfig::default();
//!     let mut pipeline = StreamingPipeline::new(
//!         Path::new("/path/to/frames"),
//!         config,
//!     ).await?;
//!
//!     pipeline.start().await?;
//!
//!     // Consume frames
//!     while let Some(frame) = pipeline.next_frame().await {
//!         println!("Got frame: {:?}", frame.frame_id);
//!     }
//!
//!     Ok(())
//! }
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::RwLock;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{error, info, trace, warn};

use crate::buffer::{BufferStats, FrameConsumer, FrameRingBuffer};
use crate::sync::FrameTelemetrySync;
use crate::watcher::{FrameEvent, FrameWatcher};
use drivedit_io::decode_bmp;
use drivedit_types::{DriveditError, RawFrame, Result, StreamConfig, Telemetry};

/// Pipeline status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineStatus {
    /// Pipeline is idle (not started).
    Idle,
    /// Pipeline is running.
    Running,
    /// Pipeline is stopped.
    Stopped,
    /// Pipeline encountered an error.
    Error,
}

/// Streaming pipeline that combines all components.
pub struct StreamingPipeline {
    /// Configuration.
    config: StreamConfig,
    /// Watch directory path.
    watch_path: PathBuf,
    /// Frame consumer for receiving frames.
    consumer: Option<FrameConsumer>,
    /// Buffer statistics.
    stats: Option<Arc<BufferStats>>,
    /// Frame-telemetry synchronizer.
    sync: Arc<FrameTelemetrySync>,
    /// Pipeline status.
    status: Arc<RwLock<PipelineStatus>>,
    /// Shutdown signal sender.
    shutdown_tx: Option<mpsc::Sender<()>>,
}

impl StreamingPipeline {
    /// Create a new streaming pipeline.
    ///
    /// # Arguments
    ///
    /// * `watch_path` - Directory to watch for new frames
    /// * `config` - Pipeline configuration
    pub async fn new(watch_path: impl AsRef<Path>, config: StreamConfig) -> Result<Self> {
        let watch_path = watch_path.as_ref().to_path_buf();

        // Verify path exists
        if !watch_path.exists() {
            return Err(DriveditError::Frame(format!(
                "Watch path does not exist: {:?}",
                watch_path
            )));
        }

        let sync = Arc::new(FrameTelemetrySync::new(Duration::from_millis(
            config.timeout_ms,
        )));

        Ok(Self {
            config,
            watch_path,
            consumer: None,
            stats: None,
            sync,
            status: Arc::new(RwLock::new(PipelineStatus::Idle)),
            shutdown_tx: None,
        })
    }

    /// Start the streaming pipeline.
    ///
    /// This spawns background tasks for:
    /// - File watching
    /// - Frame decoding
    /// - Timeout cleanup
    pub async fn start(&mut self) -> Result<()> {
        if *self.status.read() == PipelineStatus::Running {
            return Ok(());
        }

        // Create the ring buffer
        let (producer, consumer) = FrameRingBuffer::new(self.config.buffer_capacity);
        let stats = producer.stats().clone();

        self.consumer = Some(consumer);
        self.stats = Some(stats.clone());

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Create the watcher
        let mut watcher = FrameWatcher::new(&self.watch_path).await?;

        // Spawn the processing task
        let status = self.status.clone();
        let sync = self.sync.clone();
        let _timeout_ms = self.config.timeout_ms;
        let watch_path = self.watch_path.clone();

        *status.write() = PipelineStatus::Running;
        info!("Starting streaming pipeline for {:?}", watch_path);

        tokio::spawn(async move {
            let mut producer = producer;
            let mut cleanup_interval = interval(Duration::from_secs(1));

            loop {
                tokio::select! {
                    // Handle shutdown
                    _ = shutdown_rx.recv() => {
                        info!("Streaming pipeline shutting down");
                        *status.write() = PipelineStatus::Stopped;
                        break;
                    }

                    // Process file events
                    event = watcher.next() => {
                        match event {
                            Some(FrameEvent::Created { frame_id, path }) => {
                                match Self::process_frame(&path, frame_id, &sync).await {
                                    Ok(Some(frame)) => {
                                        if !producer.push(frame) {
                                            warn!("Buffer full, frame {} dropped", frame_id);
                                        }
                                    }
                                    Ok(None) => {
                                        // Frame waiting for telemetry
                                        trace!("Frame {} waiting for telemetry", frame_id);
                                    }
                                    Err(e) => {
                                        error!("Failed to process frame {}: {}", frame_id, e);
                                    }
                                }
                            }
                            Some(FrameEvent::Error(e)) => {
                                error!("Watcher error: {}", e);
                            }
                            None => {
                                warn!("Watcher channel closed");
                                *status.write() = PipelineStatus::Error;
                                break;
                            }
                        }
                    }

                    // Periodic cleanup
                    _ = cleanup_interval.tick() => {
                        sync.cleanup_timeouts();
                    }
                }
            }
        });

        Ok(())
    }

    /// Process a single frame file.
    async fn process_frame(
        path: &Path,
        frame_id: u64,
        _sync: &FrameTelemetrySync,
    ) -> Result<Option<RawFrame>> {
        // Read the file
        let bytes = tokio::fs::read(path).await?;

        // Decode the BMP
        let raw_frame = decode_bmp(&bytes)?;

        // Create frame with metadata
        // The decoded frame has default frame_id of 0, so we update it
        let frame = RawFrame::new(
            frame_id,
            raw_frame.timestamp_ms,
            raw_frame.pixels,
            raw_frame.width,
            raw_frame.height,
            raw_frame.telemetry,
        );

        // For now, return the frame directly without sync
        // In a full implementation, this would wait for telemetry
        Ok(Some(frame))
    }

    /// Stop the streaming pipeline.
    pub async fn stop(&mut self) -> Result<()> {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }
        *self.status.write() = PipelineStatus::Stopped;
        info!("Streaming pipeline stopped");
        Ok(())
    }

    /// Get the next frame from the buffer.
    ///
    /// Returns `None` if no frame is available.
    pub fn next_frame(&mut self) -> Option<RawFrame> {
        self.consumer.as_mut()?.pop()
    }

    /// Get a batch of frames.
    ///
    /// Returns up to `max_count` frames.
    pub fn next_batch(&mut self, max_count: usize) -> Vec<RawFrame> {
        match self.consumer.as_mut() {
            Some(consumer) => consumer.pop_batch(max_count),
            None => Vec::new(),
        }
    }

    /// Add telemetry for a frame.
    ///
    /// This may trigger synchronization if the corresponding frame
    /// is already in the buffer.
    pub fn add_telemetry(&self, frame_id: u64, timestamp_ms: f64, telemetry: Telemetry) {
        self.sync.add_telemetry(frame_id, timestamp_ms, telemetry);
    }

    /// Get the current pipeline status.
    pub fn status(&self) -> PipelineStatus {
        *self.status.read()
    }

    /// Check if the pipeline is running.
    pub fn is_running(&self) -> bool {
        self.status() == PipelineStatus::Running
    }

    /// Get buffer statistics.
    pub fn stats(&self) -> Option<&Arc<BufferStats>> {
        self.stats.as_ref()
    }

    /// Get the number of frames currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.consumer.as_ref().map(|c| c.len()).unwrap_or(0)
    }

    /// Get pending sync counts.
    pub fn pending_sync_counts(&self) -> (usize, usize) {
        (
            self.sync.pending_frames_count(),
            self.sync.pending_telemetry_count(),
        )
    }

    /// Get configuration.
    pub fn config(&self) -> &StreamConfig {
        &self.config
    }
}

impl Drop for StreamingPipeline {
    fn drop(&mut self) {
        // Send shutdown signal (non-blocking)
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.try_send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let temp_dir = TempDir::new().unwrap();
        let config = StreamConfig::default();

        let result = StreamingPipeline::new(temp_dir.path(), config).await;
        assert!(result.is_ok());

        let pipeline = result.unwrap();
        assert_eq!(pipeline.status(), PipelineStatus::Idle);
    }

    #[tokio::test]
    async fn test_pipeline_invalid_path() {
        let config = StreamConfig::default();
        let result = StreamingPipeline::new("/nonexistent/path", config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_start_stop() {
        let temp_dir = TempDir::new().unwrap();
        let config = StreamConfig::default();

        let mut pipeline = StreamingPipeline::new(temp_dir.path(), config)
            .await
            .unwrap();

        assert_eq!(pipeline.status(), PipelineStatus::Idle);

        pipeline.start().await.unwrap();
        assert_eq!(pipeline.status(), PipelineStatus::Running);

        pipeline.stop().await.unwrap();
        assert_eq!(pipeline.status(), PipelineStatus::Stopped);
    }

    #[tokio::test]
    async fn test_pipeline_double_start() {
        let temp_dir = TempDir::new().unwrap();
        let config = StreamConfig::default();

        let mut pipeline = StreamingPipeline::new(temp_dir.path(), config)
            .await
            .unwrap();

        pipeline.start().await.unwrap();
        // Second start should be a no-op
        pipeline.start().await.unwrap();
        assert_eq!(pipeline.status(), PipelineStatus::Running);
    }

    #[test]
    fn test_pipeline_status_enum() {
        assert_ne!(PipelineStatus::Idle, PipelineStatus::Running);
        assert_eq!(PipelineStatus::Idle, PipelineStatus::Idle);
    }
}
