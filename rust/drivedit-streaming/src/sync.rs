//! Frame-telemetry synchronization.
//!
//! This module provides synchronization between video frames and telemetry
//! data, matching them by frame ID with timeout handling for unmatched entries.
//!
//! # Example
//!
//! ```rust
//! use drivedit_streaming::sync::FrameTelemetrySync;
//! use drivedit_types::{RawFrame, Telemetry, FrameMetadata};
//! use std::time::Duration;
//!
//! let mut sync = FrameTelemetrySync::new(Duration::from_secs(1));
//!
//! // Add telemetry
//! let telemetry = Telemetry {
//!     speed_kmh: 60.0,
//!     ..Default::default()
//! };
//! sync.add_telemetry(42, 1000.0, telemetry);
//!
//! // Add frame
//! let frame = RawFrame::new(42, 0.0, vec![0u8; 100], 10, 10, Telemetry::default());
//! let metadata = FrameMetadata { frame_id: 42, ..Default::default() };
//!
//! // Try to sync
//! if let Some((synced_frame, synced_telemetry)) = sync.try_sync(frame, &metadata) {
//!     println!("Synced frame {} with telemetry", metadata.frame_id);
//! }
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use tracing::{debug, trace};

use drivedit_types::{FrameMetadata, RawFrame, Telemetry};

/// Entry in the pending queue.
#[derive(Debug)]
struct PendingEntry<T> {
    /// The data waiting to be matched.
    data: T,
    /// When this entry was added.
    added_at: Instant,
}

/// Synchronizer for matching frames with telemetry by frame_id.
///
/// Both frames and telemetry can arrive in any order. The synchronizer
/// maintains pending queues for unmatched entries and times them out
/// after a configurable duration.
pub struct FrameTelemetrySync {
    /// Pending frames waiting for telemetry.
    pending_frames: Mutex<HashMap<u64, PendingEntry<RawFrame>>>,
    /// Pending telemetry waiting for frames.
    pending_telemetry: Mutex<HashMap<u64, PendingEntry<(f64, Telemetry)>>>,
    /// Timeout for pending entries.
    timeout: Duration,
    /// Statistics.
    stats: SyncStats,
}

/// Statistics for synchronization.
#[derive(Debug, Default)]
pub struct SyncStats {
    /// Successfully synced pairs.
    pub synced: std::sync::atomic::AtomicU64,
    /// Frames that timed out.
    pub frames_timeout: std::sync::atomic::AtomicU64,
    /// Telemetry that timed out.
    pub telemetry_timeout: std::sync::atomic::AtomicU64,
}

impl SyncStats {
    /// Get synced count.
    pub fn synced(&self) -> u64 {
        self.synced.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get frame timeout count.
    pub fn frames_timeout(&self) -> u64 {
        self.frames_timeout
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Get telemetry timeout count.
    pub fn telemetry_timeout(&self) -> u64 {
        self.telemetry_timeout
            .load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl FrameTelemetrySync {
    /// Create a new synchronizer with the given timeout.
    pub fn new(timeout: Duration) -> Self {
        Self {
            pending_frames: Mutex::new(HashMap::new()),
            pending_telemetry: Mutex::new(HashMap::new()),
            timeout,
            stats: SyncStats::default(),
        }
    }

    /// Create with default timeout (1 second).
    pub fn with_default_timeout() -> Self {
        Self::new(Duration::from_secs(1))
    }

    /// Add telemetry data for a frame.
    ///
    /// If a matching frame is already pending, returns the synced pair.
    /// Otherwise, adds to the pending telemetry queue.
    pub fn add_telemetry(
        &self,
        frame_id: u64,
        timestamp_ms: f64,
        telemetry: Telemetry,
    ) -> Option<(RawFrame, Telemetry)> {
        let now = Instant::now();

        // First, check if we have a pending frame
        {
            let mut pending_frames = self.pending_frames.lock();
            if let Some(entry) = pending_frames.remove(&frame_id) {
                self.stats
                    .synced
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                debug!("Synced frame {} (frame was pending)", frame_id);
                return Some((entry.data, telemetry));
            }
        }

        // No pending frame, add to pending telemetry
        {
            let mut pending_telemetry = self.pending_telemetry.lock();
            pending_telemetry.insert(
                frame_id,
                PendingEntry {
                    data: (timestamp_ms, telemetry),
                    added_at: now,
                },
            );
            trace!("Added pending telemetry for frame {}", frame_id);
        }

        None
    }

    /// Try to sync a frame with its telemetry.
    ///
    /// If matching telemetry is available, returns the synced pair.
    /// Otherwise, adds the frame to the pending queue.
    pub fn try_sync(
        &self,
        frame: RawFrame,
        metadata: &FrameMetadata,
    ) -> Option<(RawFrame, Telemetry)> {
        let frame_id = metadata.frame_id;
        let now = Instant::now();

        // Check if we have pending telemetry
        {
            let mut pending_telemetry = self.pending_telemetry.lock();
            if let Some(entry) = pending_telemetry.remove(&frame_id) {
                self.stats
                    .synced
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                debug!("Synced frame {} (telemetry was pending)", frame_id);
                return Some((frame, entry.data.1));
            }
        }

        // No pending telemetry, add to pending frames
        {
            let mut pending_frames = self.pending_frames.lock();
            pending_frames.insert(
                frame_id,
                PendingEntry {
                    data: frame,
                    added_at: now,
                },
            );
            trace!("Added pending frame {}", frame_id);
        }

        None
    }

    /// Add a frame without syncing immediately.
    ///
    /// The frame will wait for telemetry to arrive.
    pub fn add_frame(&self, frame_id: u64, frame: RawFrame) {
        let now = Instant::now();
        let mut pending_frames = self.pending_frames.lock();
        pending_frames.insert(
            frame_id,
            PendingEntry {
                data: frame,
                added_at: now,
            },
        );
    }

    /// Clean up timed-out entries.
    ///
    /// Returns the number of entries removed.
    pub fn cleanup_timeouts(&self) -> usize {
        let now = Instant::now();
        let mut removed = 0;

        // Clean pending frames
        {
            let mut pending_frames = self.pending_frames.lock();
            let initial_len = pending_frames.len();
            pending_frames.retain(|_, entry| {
                let expired = now.duration_since(entry.added_at) > self.timeout;
                if expired {
                    self.stats
                        .frames_timeout
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                !expired
            });
            removed += initial_len - pending_frames.len();
        }

        // Clean pending telemetry
        {
            let mut pending_telemetry = self.pending_telemetry.lock();
            let initial_len = pending_telemetry.len();
            pending_telemetry.retain(|_, entry| {
                let expired = now.duration_since(entry.added_at) > self.timeout;
                if expired {
                    self.stats
                        .telemetry_timeout
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                !expired
            });
            removed += initial_len - pending_telemetry.len();
        }

        if removed > 0 {
            debug!("Cleaned up {} timed-out sync entries", removed);
        }

        removed
    }

    /// Get the number of pending frames.
    pub fn pending_frames_count(&self) -> usize {
        self.pending_frames.lock().len()
    }

    /// Get the number of pending telemetry entries.
    pub fn pending_telemetry_count(&self) -> usize {
        self.pending_telemetry.lock().len()
    }

    /// Get synchronization statistics.
    pub fn stats(&self) -> &SyncStats {
        &self.stats
    }

    /// Clear all pending entries.
    pub fn clear(&self) {
        self.pending_frames.lock().clear();
        self.pending_telemetry.lock().clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_frame(id: u64) -> RawFrame {
        RawFrame::new(id, 0.0, vec![0u8; 100], 10, 10, Telemetry::default())
    }

    fn make_test_telemetry() -> Telemetry {
        Telemetry {
            speed_kmh: 60.0,
            steering: 0.1,
            ..Default::default()
        }
    }

    #[test]
    fn test_sync_creation() {
        let sync = FrameTelemetrySync::new(Duration::from_secs(1));
        assert_eq!(sync.pending_frames_count(), 0);
        assert_eq!(sync.pending_telemetry_count(), 0);
    }

    #[test]
    fn test_frame_first_then_telemetry() {
        let sync = FrameTelemetrySync::new(Duration::from_secs(1));

        // Add frame first
        let frame = make_test_frame(42);
        let metadata = FrameMetadata {
            frame_id: 42,
            ..Default::default()
        };
        assert!(sync.try_sync(frame, &metadata).is_none());
        assert_eq!(sync.pending_frames_count(), 1);

        // Add telemetry - should sync
        let telemetry = make_test_telemetry();
        let result = sync.add_telemetry(42, 1000.0, telemetry);
        assert!(result.is_some());

        let (synced_frame, synced_telemetry) = result.unwrap();
        assert_eq!(synced_frame.frame_id, 42);
        assert_eq!(synced_telemetry.speed_kmh, 60.0);

        assert_eq!(sync.pending_frames_count(), 0);
        assert_eq!(sync.stats().synced(), 1);
    }

    #[test]
    fn test_telemetry_first_then_frame() {
        let sync = FrameTelemetrySync::new(Duration::from_secs(1));

        // Add telemetry first
        let telemetry = make_test_telemetry();
        assert!(sync.add_telemetry(42, 1000.0, telemetry).is_none());
        assert_eq!(sync.pending_telemetry_count(), 1);

        // Add frame - should sync
        let frame = make_test_frame(42);
        let metadata = FrameMetadata {
            frame_id: 42,
            ..Default::default()
        };
        let result = sync.try_sync(frame, &metadata);
        assert!(result.is_some());

        let (synced_frame, synced_telemetry) = result.unwrap();
        assert_eq!(synced_frame.frame_id, 42);
        assert_eq!(synced_telemetry.speed_kmh, 60.0);

        assert_eq!(sync.pending_telemetry_count(), 0);
        assert_eq!(sync.stats().synced(), 1);
    }

    #[test]
    fn test_timeout_cleanup() {
        let sync = FrameTelemetrySync::new(Duration::from_millis(10));

        // Add entries
        sync.add_frame(1, make_test_frame(1));
        sync.add_telemetry(2, 1000.0, make_test_telemetry());

        assert_eq!(sync.pending_frames_count(), 1);
        assert_eq!(sync.pending_telemetry_count(), 1);

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(20));

        // Cleanup should remove them
        let removed = sync.cleanup_timeouts();
        assert_eq!(removed, 2);

        assert_eq!(sync.pending_frames_count(), 0);
        assert_eq!(sync.pending_telemetry_count(), 0);
        assert_eq!(sync.stats().frames_timeout(), 1);
        assert_eq!(sync.stats().telemetry_timeout(), 1);
    }

    #[test]
    fn test_clear() {
        let sync = FrameTelemetrySync::new(Duration::from_secs(1));

        sync.add_frame(1, make_test_frame(1));
        sync.add_frame(2, make_test_frame(2));
        sync.add_telemetry(3, 1000.0, make_test_telemetry());

        sync.clear();

        assert_eq!(sync.pending_frames_count(), 0);
        assert_eq!(sync.pending_telemetry_count(), 0);
    }
}
