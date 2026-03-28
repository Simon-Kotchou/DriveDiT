//! Enfusion session management for training data access.
//!
//! An EnfusionSession represents a single recording session with
//! synchronized video frames, telemetry, and optional depth/flow data.

use crate::depth::DepthLoader;
use crate::error::{BatchError, Result};
use crate::loader::FrameLoader;
use drivedit_io::{ENFCAPReader, TelemetryParser};
use drivedit_types::{BatchConfig, Telemetry};
use ndarray::{Array2, Array4};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Metadata about a session.
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    /// Session identifier.
    pub session_id: String,
    /// Total number of frames.
    pub num_frames: usize,
    /// Duration in seconds.
    pub duration_seconds: f64,
    /// Frames per second.
    pub fps: f32,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Whether telemetry is available.
    pub has_telemetry: bool,
    /// Whether depth data is available.
    pub has_depth: bool,
    /// Session path.
    pub path: String,
}

impl std::fmt::Display for SessionMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SessionMetadata(id='{}', frames={}, duration={:.1}s, {}x{})",
            self.session_id, self.num_frames, self.duration_seconds, self.width, self.height
        )
    }
}

/// A training sample from a session.
#[derive(Debug, Clone)]
pub struct SessionSample {
    /// Frame data [T, C, H, W].
    pub frames: Array4<f32>,
    /// Control signals [T, control_dim].
    pub controls: Array2<f32>,
    /// Depth data [T, 1, H, W] (optional).
    pub depth: Option<Array4<f32>>,
    /// Frame indices in the session.
    pub frame_indices: Vec<usize>,
    /// Session identifier.
    pub session_id: String,
}

/// Manages a single Enfusion recording session.
pub struct EnfusionSession {
    /// Session configuration.
    config: BatchConfig,
    /// Session root path.
    #[allow(dead_code)]
    path: PathBuf,
    /// Session identifier.
    session_id: String,
    /// Frame loader.
    frame_loader: FrameLoader,
    /// Depth loader (optional).
    depth_loader: Option<DepthLoader>,
    /// Telemetry parser.
    telemetry_parser: TelemetryParser,
    /// ENFCAP reader (for binary format).
    enfcap_reader: Option<ENFCAPReader>,
    /// Cached telemetry data.
    telemetry_cache: Arc<RwLock<Option<HashMap<u64, (f64, Telemetry)>>>>,
    /// Frame paths (for image-based sessions).
    frame_paths: Vec<PathBuf>,
    /// Depth paths (optional).
    depth_paths: Vec<PathBuf>,
    /// Session metadata.
    metadata: SessionMetadata,
}

impl EnfusionSession {
    /// Create a session from a directory path.
    pub fn from_path(path: &Path, config: BatchConfig) -> Result<Self> {
        if !path.exists() {
            return Err(BatchError::Session(format!(
                "Session path does not exist: {:?}",
                path
            )));
        }

        let session_id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Detect session type and find data files
        let (frame_paths, enfcap_reader, _has_enfcap) = Self::detect_frames(path)?;
        let depth_paths = Self::find_depth_files(path);
        let has_depth = !depth_paths.is_empty() && config.load_depth;
        let has_telemetry = Self::find_telemetry_file(path).is_some();

        // Get frame count and dimensions
        let (num_frames, width, height, fps) = if let Some(ref reader) = enfcap_reader {
            let dims = reader.dimensions();
            (reader.frame_count() as usize, dims.0, dims.1, reader.header().fps)
        } else {
            // Probe first image for dimensions
            let (w, h) = if !frame_paths.is_empty() {
                Self::probe_image_dimensions(&frame_paths[0])?
            } else {
                return Err(BatchError::Session("No frames found".to_string()));
            };
            (frame_paths.len(), w, h, 30.0_f32) // Assume 30fps for images
        };

        let duration_seconds = num_frames as f64 / fps as f64;

        let metadata = SessionMetadata {
            session_id: session_id.clone(),
            num_frames,
            duration_seconds,
            fps,
            width,
            height,
            has_telemetry,
            has_depth,
            path: path.to_string_lossy().to_string(),
        };

        let frame_loader = FrameLoader::new(config.clone());
        let depth_loader = if has_depth {
            Some(DepthLoader::new(crate::depth::DepthLoaderConfig::with_resolution(
                config.target_width,
                config.target_height,
            )))
        } else {
            None
        };

        let telemetry_path = Self::find_telemetry_file(path);
        let telemetry_parser = TelemetryParser::new(&telemetry_path.unwrap_or_else(|| path.join("telemetry.csv")));

        Ok(Self {
            config,
            path: path.to_path_buf(),
            session_id,
            frame_loader,
            depth_loader,
            telemetry_parser,
            enfcap_reader,
            telemetry_cache: Arc::new(RwLock::new(None)),
            frame_paths,
            depth_paths,
            metadata,
        })
    }

    /// Get session metadata.
    pub fn metadata(&self) -> &SessionMetadata {
        &self.metadata
    }

    /// Get the number of frames in the session.
    pub fn num_frames(&self) -> usize {
        self.metadata.num_frames
    }

    /// Get the number of valid samples that can be extracted.
    pub fn num_samples(&self) -> usize {
        let total_needed = self.config.total_frames() * self.config.frame_stride;
        if self.metadata.num_frames < total_needed {
            0
        } else {
            (self.metadata.num_frames - total_needed) / self.config.frame_stride + 1
        }
    }

    /// Get frames for a sample at the given index.
    pub fn get_frames(&self, sample_idx: usize) -> Result<Array4<f32>> {
        self.load_sample_frames(sample_idx)
    }

    /// Get control signals for a sample at the given index.
    pub fn get_controls(&mut self, sample_idx: usize) -> Result<Array2<f32>> {
        self.load_sample_controls(sample_idx)
    }

    /// Get depth data for a sample (if available).
    pub fn get_depth(&self, sample_idx: usize) -> Result<Option<Array4<f32>>> {
        if !self.metadata.has_depth || self.depth_loader.is_none() {
            return Ok(None);
        }
        let depth = self.load_sample_depth(sample_idx)?;
        Ok(Some(depth))
    }

    /// Get a complete sample (frames + controls + optional depth).
    pub fn get_sample(&mut self, sample_idx: usize) -> Result<SessionSample> {
        let indices = self.sample_frame_indices(sample_idx)?;
        let frames = self.load_sample_frames(sample_idx)?;
        let controls = self.load_sample_controls(sample_idx)?;
        let depth = if self.metadata.has_depth {
            Some(self.load_sample_depth(sample_idx)?)
        } else {
            None
        };

        Ok(SessionSample {
            frames,
            controls,
            depth,
            frame_indices: indices,
            session_id: self.session_id.clone(),
        })
    }

    /// Preload telemetry data into cache.
    pub fn preload_telemetry(&mut self) -> Result<()> {
        self.load_telemetry()?;
        Ok(())
    }

    /// Preload frame data into cache.
    pub fn preload_frames(&self, start: usize, end: usize) -> Result<usize> {
        let paths: Vec<String> = self.frame_paths[start..end.min(self.frame_paths.len())]
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        self.frame_loader.preload(paths)
    }

    /// Clear all caches.
    pub fn clear_cache(&mut self) {
        self.frame_loader.clear_cache();
        *self.telemetry_cache.write() = None;
    }

    /// Detect frame files in a session directory.
    fn detect_frames(
        path: &Path,
    ) -> Result<(Vec<PathBuf>, Option<ENFCAPReader>, bool)> {
        // Check for ENFCAP files first
        let enfcap_path = path.join("video.enfcap");
        if enfcap_path.exists() {
            let reader = ENFCAPReader::open(&enfcap_path)
                .map_err(|e| BatchError::Session(format!("Failed to open ENFCAP: {}", e)))?;
            return Ok((Vec::new(), Some(reader), true));
        }

        // Look for image files
        let frames_dir = if path.join("frames").exists() {
            path.join("frames")
        } else if path.join("images").exists() {
            path.join("images")
        } else {
            path.to_path_buf()
        };

        let mut frame_paths: Vec<PathBuf> = fs::read_dir(&frames_dir)
            .map_err(|e| BatchError::Session(format!("Cannot read frames dir: {}", e)))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp" | "bmp"))
                    .unwrap_or(false)
            })
            .collect();

        frame_paths.sort();

        if frame_paths.is_empty() {
            return Err(BatchError::Session(format!(
                "No frame files found in {:?}",
                frames_dir
            )));
        }

        Ok((frame_paths, None, false))
    }

    /// Find depth files in a session directory.
    fn find_depth_files(path: &Path) -> Vec<PathBuf> {
        let depth_dir = if path.join("depth").exists() {
            path.join("depth")
        } else {
            return Vec::new();
        };

        let mut depth_paths: Vec<PathBuf> = fs::read_dir(&depth_dir)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension()
                            .and_then(|e| e.to_str())
                            .map(|e| matches!(e.to_lowercase().as_str(), "png" | "npz" | "npy"))
                            .unwrap_or(false)
                    })
                    .collect()
            })
            .unwrap_or_default();

        depth_paths.sort();
        depth_paths
    }

    /// Find telemetry file in a session directory.
    fn find_telemetry_file(path: &Path) -> Option<PathBuf> {
        let candidates = [
            "telemetry.csv",
            "controls.csv",
            "data.csv",
            "log.csv",
        ];

        for name in &candidates {
            let file_path = path.join(name);
            if file_path.exists() {
                return Some(file_path);
            }
        }

        None
    }

    /// Probe image dimensions from a file.
    fn probe_image_dimensions(path: &Path) -> Result<(u32, u32)> {
        let reader = image::ImageReader::open(path)
            .map_err(|e| BatchError::Frame(format!("Failed to open image: {}", e)))?;
        let dims = reader
            .into_dimensions()
            .map_err(|e| BatchError::Frame(format!("Failed to read dimensions: {}", e)))?;
        Ok(dims)
    }

    /// Calculate frame indices for a sample.
    fn sample_frame_indices(&self, sample_idx: usize) -> Result<Vec<usize>> {
        let total_frames = self.config.total_frames();
        let stride = self.config.frame_stride;
        let start = sample_idx * stride;

        if start + total_frames * stride > self.metadata.num_frames {
            return Err(BatchError::IndexOutOfBounds {
                index: sample_idx,
                length: self.num_samples(),
            });
        }

        let indices: Vec<usize> = (0..total_frames).map(|i| start + i * stride).collect();
        Ok(indices)
    }

    /// Load frames for a sample.
    fn load_sample_frames(&self, sample_idx: usize) -> Result<Array4<f32>> {
        let indices = self.sample_frame_indices(sample_idx)?;

        if self.enfcap_reader.is_some() {
            return Err(BatchError::Frame("ENFCAP reading not yet implemented".to_string()));
        }

        let paths: Vec<String> = indices
            .iter()
            .map(|&i| self.frame_paths[i].to_string_lossy().to_string())
            .collect();
        self.frame_loader.load_frames_array(&paths)
    }

    /// Load controls for a sample.
    fn load_sample_controls(&mut self, sample_idx: usize) -> Result<Array2<f32>> {
        let telemetry = self.load_telemetry()?;
        let indices = self.sample_frame_indices(sample_idx)?;

        let total_frames = indices.len();
        let control_dim = 6; // 6D control: steer, accel, goal_x, goal_y, speed, heading_rate

        let mut controls = Array2::zeros((total_frames, control_dim));

        for (i, &frame_idx) in indices.iter().enumerate() {
            if let Some((_ts, tel)) = telemetry.get(&(frame_idx as u64)) {
                let ctrl = tel.to_control_array();
                for (j, &val) in ctrl.iter().enumerate() {
                    if j < control_dim {
                        controls[[i, j]] = val;
                    }
                }
            }
        }

        Ok(controls)
    }

    /// Load depth data for a sample.
    fn load_sample_depth(&self, sample_idx: usize) -> Result<Array4<f32>> {
        let depth_loader = self
            .depth_loader
            .as_ref()
            .ok_or_else(|| BatchError::Depth("Depth loader not available".to_string()))?;

        let indices = self.sample_frame_indices(sample_idx)?;

        let paths: Vec<&Path> = indices
            .iter()
            .filter_map(|&i| self.depth_paths.get(i).map(|p| p.as_path()))
            .collect();

        if paths.len() != indices.len() {
            return Err(BatchError::Depth("Missing depth files".to_string()));
        }

        let (data, batch_size, height, width) = depth_loader.load_batch_contiguous(&paths)
            .map_err(|e| BatchError::Depth(e.to_string()))?;

        // Reshape to [T, 1, H, W]
        let mut array = Array4::zeros((batch_size, 1, height, width));
        for t in 0..batch_size {
            for y in 0..height {
                for x in 0..width {
                    let idx = t * height * width + y * width + x;
                    array[[t, 0, y, x]] = data[idx];
                }
            }
        }

        Ok(array)
    }

    /// Load or retrieve cached telemetry data.
    fn load_telemetry(&mut self) -> Result<HashMap<u64, (f64, Telemetry)>> {
        // Check cache
        {
            let cache = self.telemetry_cache.read();
            if let Some(ref telemetry) = *cache {
                return Ok(telemetry.clone());
            }
        }

        // Parse telemetry file
        self.telemetry_parser.parse_updates()
            .map_err(|e| BatchError::Telemetry(format!("Failed to parse telemetry: {}", e)))?;

        let telemetry = self.telemetry_parser.data().clone();

        // Cache
        {
            let mut cache = self.telemetry_cache.write();
            *cache = Some(telemetry.clone());
        }

        Ok(telemetry)
    }
}

impl std::fmt::Display for EnfusionSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EnfusionSession('{}', frames={}, samples={})",
            self.session_id,
            self.num_frames(),
            self.num_samples()
        )
    }
}

/// Find all session directories in a root path.
pub fn find_sessions(root: &Path) -> Result<Vec<PathBuf>> {
    let mut sessions = Vec::new();

    if !root.exists() {
        return Err(BatchError::Session(format!(
            "Root path does not exist: {:?}",
            root
        )));
    }

    // Check if root itself is a session
    if is_session_dir(root) {
        sessions.push(root.to_path_buf());
        return Ok(sessions);
    }

    // Recursively find session directories
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            if is_session_dir(&path) {
                sessions.push(path);
            } else {
                // Check one level deeper
                if let Ok(entries) = fs::read_dir(&path) {
                    for subentry in entries.flatten() {
                        let subpath = subentry.path();
                        if subpath.is_dir() && is_session_dir(&subpath) {
                            sessions.push(subpath);
                        }
                    }
                }
            }
        }
    }

    sessions.sort();
    Ok(sessions)
}

/// Check if a directory looks like a session directory.
pub fn is_session_dir(path: &Path) -> bool {
    // Has ENFCAP file
    if path.join("video.enfcap").exists() {
        return true;
    }

    // Has frames directory or image files
    if path.join("frames").exists() || path.join("images").exists() {
        return true;
    }

    // Has image files directly
    fs::read_dir(path)
        .ok()
        .map(|entries| {
            entries.filter_map(|e| e.ok()).any(|e| {
                e.path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_metadata_display() {
        let meta = SessionMetadata {
            session_id: "test".to_string(),
            num_frames: 100,
            duration_seconds: 10.0,
            fps: 10.0,
            width: 640,
            height: 360,
            has_telemetry: true,
            has_depth: false,
            path: "/test/path".to_string(),
        };

        let display = format!("{}", meta);
        assert!(display.contains("test"));
        assert!(display.contains("100"));
    }

    #[test]
    fn test_is_session_dir_with_enfcap() {
        // Would need temp directory setup for proper testing
    }

    #[test]
    fn test_sample_indices_calculation() {
        let _config = BatchConfig {
            sequence_length: 8,
            context_frames: 2,
            frame_stride: 2,
            ..BatchConfig::default()
        };

        // Total frames needed = (8 + 2) * 2 = 20
        // With 100 frames, samples = (100 - 20) / 2 + 1 = 41
    }
}
