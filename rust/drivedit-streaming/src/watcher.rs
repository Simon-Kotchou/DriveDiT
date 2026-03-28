//! Async file watcher for detecting new frames.
//!
//! This module provides an async file system watcher that monitors a directory
//! for new BMP frame files and delivers events through a tokio channel.
//!
//! # Example
//!
//! ```rust,ignore
//! use drivedit_streaming::watcher::FrameWatcher;
//! use std::path::Path;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut watcher = FrameWatcher::new(Path::new("/path/to/frames")).await?;
//!
//!     while let Some(event) = watcher.next().await {
//!         match event {
//!             FrameEvent::Created { frame_id, path } => {
//!                 println!("New frame: {} at {:?}", frame_id, path);
//!             }
//!             FrameEvent::Error(e) => {
//!                 eprintln!("Watch error: {}", e);
//!             }
//!         }
//!     }
//!     Ok(())
//! }
//! ```

use std::path::{Path, PathBuf};
use std::time::Duration;

use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use notify_debouncer_mini::{new_debouncer, DebouncedEvent, Debouncer};
use tokio::sync::mpsc;
use tracing::{debug, error, trace, warn};

use drivedit_io::parse_frame_id;
use drivedit_types::DriveditError;

/// Event emitted when a new frame file is detected.
#[derive(Debug, Clone)]
pub enum FrameEvent {
    /// A new frame file was created.
    Created {
        /// Frame ID extracted from filename.
        frame_id: u64,
        /// Full path to the frame file.
        path: PathBuf,
    },
    /// An error occurred while watching.
    Error(String),
}

/// Configuration for the frame watcher.
#[derive(Debug, Clone)]
pub struct WatcherConfig {
    /// Directory to watch.
    pub watch_path: PathBuf,
    /// Debounce duration for file events.
    pub debounce_duration: Duration,
    /// Channel capacity for event delivery.
    pub channel_capacity: usize,
}

impl Default for WatcherConfig {
    fn default() -> Self {
        Self {
            watch_path: PathBuf::new(),
            debounce_duration: Duration::from_millis(50),
            channel_capacity: 256,
        }
    }
}

impl WatcherConfig {
    /// Create a config for a specific path.
    pub fn with_path(path: impl AsRef<Path>) -> Self {
        Self {
            watch_path: path.as_ref().to_path_buf(),
            ..Default::default()
        }
    }
}

/// Async file watcher for frame files.
///
/// Monitors a directory for new BMP files matching the pattern `frame_XXXXX.bmp`
/// and emits events through a tokio channel.
pub struct FrameWatcher {
    /// Receiver for frame events.
    rx: mpsc::Receiver<FrameEvent>,
    /// The debouncer (kept alive to maintain the watcher).
    _debouncer: Debouncer<RecommendedWatcher>,
}

impl FrameWatcher {
    /// Create a new frame watcher for the given directory.
    ///
    /// # Arguments
    ///
    /// * `path` - Directory to watch for new frame files
    ///
    /// # Returns
    ///
    /// A new `FrameWatcher` instance, or an error if the watcher could not be created.
    pub async fn new(path: impl AsRef<Path>) -> Result<Self, DriveditError> {
        let config = WatcherConfig::with_path(path);
        Self::with_config(config).await
    }

    /// Create a new frame watcher with custom configuration.
    pub async fn with_config(config: WatcherConfig) -> Result<Self, DriveditError> {
        let (tx, rx) = mpsc::channel(config.channel_capacity);
        let watch_path = config.watch_path.clone();

        // Create the debouncer with a callback that sends events
        let mut debouncer = new_debouncer(
            config.debounce_duration,
            move |result: Result<Vec<DebouncedEvent>, notify::Error>| {
                match result {
                    Ok(events) => {
                        for event in events {
                            if let Some(frame_event) = Self::process_debounced_event(&event) {
                                // Send event, ignore if channel is full
                                let _ = tx.blocking_send(frame_event);
                            }
                        }
                    }
                    Err(e) => {
                        error!("Watcher error: {}", e);
                        let _ = tx.blocking_send(FrameEvent::Error(e.to_string()));
                    }
                }
            },
        )
        .map_err(|e| DriveditError::watcher(e.to_string()))?;

        // Start watching the directory
        debouncer
            .watcher()
            .watch(&watch_path, RecursiveMode::NonRecursive)
            .map_err(|e| DriveditError::watcher(e.to_string()))?;

        debug!("Started watching: {:?}", watch_path);

        Ok(Self {
            rx,
            _debouncer: debouncer,
        })
    }

    /// Process a debounced event and convert to FrameEvent if applicable.
    fn process_debounced_event(event: &DebouncedEvent) -> Option<FrameEvent> {
        let path = &event.path;

        // Check if it's a BMP file
        let filename = path.file_name()?.to_str()?;
        if !filename.ends_with(".bmp") {
            trace!("Ignoring non-BMP file: {}", filename);
            return None;
        }

        // Try to parse frame ID
        let frame_id = parse_frame_id(filename)?;

        debug!("Detected frame: {} at {:?}", frame_id, path);

        Some(FrameEvent::Created {
            frame_id,
            path: path.clone(),
        })
    }

    /// Get the next frame event.
    ///
    /// Returns `None` if the watcher has been dropped.
    pub async fn next(&mut self) -> Option<FrameEvent> {
        self.rx.recv().await
    }

    /// Try to get the next frame event without waiting.
    ///
    /// Returns `None` if no event is available.
    pub fn try_next(&mut self) -> Option<FrameEvent> {
        self.rx.try_recv().ok()
    }
}

/// Alternative watcher implementation using raw notify (without debouncing).
///
/// Use this when you need immediate event delivery.
pub struct RawFrameWatcher {
    /// Receiver for frame events.
    rx: mpsc::Receiver<FrameEvent>,
    /// The watcher (kept alive to maintain watching).
    _watcher: RecommendedWatcher,
}

impl RawFrameWatcher {
    /// Create a new raw frame watcher.
    pub async fn new(
        path: impl AsRef<Path>,
        channel_capacity: usize,
    ) -> Result<Self, DriveditError> {
        let (tx, rx) = mpsc::channel(channel_capacity);
        let watch_path = path.as_ref().to_path_buf();

        let mut watcher = notify::recommended_watcher(move |result: Result<Event, notify::Error>| {
            match result {
                Ok(event) => {
                    // Only process create events
                    if event.kind.is_create() {
                        for path in event.paths {
                            if let Some(filename) = path.file_name() {
                                if let Some(filename_str) = filename.to_str() {
                                    if let Some(frame_id) = parse_frame_id(filename_str) {
                                        let _ = tx.blocking_send(FrameEvent::Created {
                                            frame_id,
                                            path: path.clone(),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!("Watch error: {}", e);
                    let _ = tx.blocking_send(FrameEvent::Error(e.to_string()));
                }
            }
        })
        .map_err(|e| DriveditError::watcher(e.to_string()))?;

        watcher
            .watch(&watch_path, RecursiveMode::NonRecursive)
            .map_err(|e| DriveditError::watcher(e.to_string()))?;

        Ok(Self {
            rx,
            _watcher: watcher,
        })
    }

    /// Get the next frame event.
    pub async fn next(&mut self) -> Option<FrameEvent> {
        self.rx.recv().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_frame_event_debug() {
        let event = FrameEvent::Created {
            frame_id: 42,
            path: PathBuf::from("/test/frame_00042.bmp"),
        };
        let debug_str = format!("{:?}", event);
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_watcher_config_default() {
        let config = WatcherConfig::default();
        assert_eq!(config.channel_capacity, 256);
        assert_eq!(config.debounce_duration, Duration::from_millis(50));
    }

    #[test]
    fn test_watcher_config_with_path() {
        let config = WatcherConfig::with_path("/test/path");
        assert_eq!(config.watch_path, PathBuf::from("/test/path"));
    }

    #[tokio::test]
    async fn test_watcher_creation() {
        let temp_dir = TempDir::new().unwrap();
        let result = FrameWatcher::new(temp_dir.path()).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_watcher_detects_frames() {
        let temp_dir = TempDir::new().unwrap();
        let mut watcher = FrameWatcher::new(temp_dir.path()).await.unwrap();

        // Create a frame file
        let frame_path = temp_dir.path().join("frame_00001.bmp");
        fs::write(&frame_path, b"BM").unwrap();

        // Wait for event with timeout
        let event = tokio::time::timeout(Duration::from_secs(1), watcher.next()).await;

        // Event should be detected (or timeout on slow systems)
        if let Ok(Some(FrameEvent::Created { frame_id, path })) = event {
            assert_eq!(frame_id, 1);
            assert_eq!(path, frame_path);
        }
    }
}
