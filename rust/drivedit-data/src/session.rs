//! Enfusion session management for training data access.
//!
//! An EnfusionSession represents a single recording session with
//! synchronized video frames, telemetry, and optional depth/flow data.

use crate::config::{DatasetConfig, TelemetryConfig};
use crate::enfcap::ENFCAPReader;
use crate::error::{DataError, DataResult};
use crate::frames::{DepthLoader, FrameLoader};
use crate::telemetry::TelemetryParser;
use ndarray::{s, Array2, Array4};
use numpy::{PyArray2, PyArray4, ToPyArray};
use parking_lot::RwLock;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Metadata about a session.
#[pyclass]
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    /// Session identifier
    #[pyo3(get)]
    pub session_id: String,
    /// Total number of frames
    #[pyo3(get)]
    pub num_frames: usize,
    /// Duration in seconds
    #[pyo3(get)]
    pub duration_seconds: f64,
    /// Frames per second
    #[pyo3(get)]
    pub fps: f32,
    /// Frame dimensions (width, height)
    #[pyo3(get)]
    pub width: u32,
    #[pyo3(get)]
    pub height: u32,
    /// Whether telemetry is available
    #[pyo3(get)]
    pub has_telemetry: bool,
    /// Whether depth data is available
    #[pyo3(get)]
    pub has_depth: bool,
    /// Session path
    #[pyo3(get)]
    pub path: String,
}

#[pymethods]
impl SessionMetadata {
    fn __repr__(&self) -> String {
        format!(
            "SessionMetadata(id='{}', frames={}, duration={:.1}s, {}x{})",
            self.session_id, self.num_frames, self.duration_seconds, self.width, self.height
        )
    }
}

/// A training sample from a session.
#[derive(Debug, Clone)]
pub struct SessionSample {
    /// Frame data [T, C, H, W]
    pub frames: Array4<f32>,
    /// Control signals [T, control_dim]
    pub controls: Array2<f32>,
    /// Depth data [T, 1, H, W] (optional)
    pub depth: Option<Array4<f32>>,
    /// Frame indices in the session
    pub frame_indices: Vec<usize>,
    /// Session identifier
    pub session_id: String,
}

/// Manages a single Enfusion recording session.
#[pyclass]
pub struct EnfusionSession {
    /// Session configuration
    config: DatasetConfig,
    /// Session root path
    path: PathBuf,
    /// Session identifier
    session_id: String,
    /// Frame loader
    frame_loader: FrameLoader,
    /// Depth loader (optional)
    depth_loader: Option<DepthLoader>,
    /// Telemetry parser
    telemetry_parser: TelemetryParser,
    /// ENFCAP reader (for binary format)
    enfcap_reader: Option<ENFCAPReader>,
    /// Cached telemetry data
    telemetry_cache: Arc<RwLock<Option<Array2<f32>>>>,
    /// Frame paths (for image-based sessions)
    frame_paths: Vec<PathBuf>,
    /// Depth paths (optional)
    depth_paths: Vec<PathBuf>,
    /// Session metadata
    metadata: SessionMetadata,
}

#[pymethods]
impl EnfusionSession {
    /// Create a new EnfusionSession from a directory path.
    #[new]
    #[pyo3(signature = (path, config = None, telemetry_config = None))]
    pub fn new(
        path: &str,
        config: Option<DatasetConfig>,
        telemetry_config: Option<TelemetryConfig>,
    ) -> PyResult<Self> {
        let config = config.unwrap_or_default();
        let telemetry_config = telemetry_config.unwrap_or_default();

        let session = Self::from_path(Path::new(path), config, telemetry_config)?;
        Ok(session)
    }

    /// Get session metadata.
    pub fn metadata(&self) -> SessionMetadata {
        self.metadata.clone()
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
    pub fn get_frames<'py>(
        &mut self,
        py: Python<'py>,
        sample_idx: usize,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let frames = self.load_sample_frames(sample_idx)?;
        Ok(frames.to_pyarray_bound(py))
    }

    /// Get control signals for a sample at the given index.
    pub fn get_controls<'py>(
        &mut self,
        py: Python<'py>,
        sample_idx: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let controls = self.load_sample_controls(sample_idx)?;
        Ok(controls.to_pyarray_bound(py))
    }

    /// Get depth data for a sample (if available).
    pub fn get_depth<'py>(
        &mut self,
        py: Python<'py>,
        sample_idx: usize,
    ) -> PyResult<Option<Bound<'py, PyArray4<f32>>>> {
        if !self.metadata.has_depth || self.depth_loader.is_none() {
            return Ok(None);
        }

        let depth = self.load_sample_depth(sample_idx)?;
        Ok(Some(depth.to_pyarray_bound(py)))
    }

    /// Get a complete sample (frames + controls + optional depth).
    pub fn get_sample<'py>(
        &mut self,
        py: Python<'py>,
        sample_idx: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )> {
        let frames = self.load_sample_frames(sample_idx)?;
        let controls = self.load_sample_controls(sample_idx)?;
        let depth = if self.metadata.has_depth {
            Some(self.load_sample_depth(sample_idx)?)
        } else {
            None
        };

        Ok((
            frames.to_pyarray_bound(py),
            controls.to_pyarray_bound(py),
            depth.map(|d| d.to_pyarray_bound(py)),
        ))
    }

    /// Preload telemetry data into cache.
    pub fn preload_telemetry(&mut self) -> PyResult<()> {
        self.load_telemetry()?;
        Ok(())
    }

    /// Preload frame data into cache.
    pub fn preload_frames(&mut self, start: usize, end: usize) -> PyResult<usize> {
        let paths: Vec<String> = self.frame_paths[start..end.min(self.frame_paths.len())]
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect();
        let loaded = self.frame_loader.preload(paths)?;
        Ok(loaded)
    }

    /// Clear all caches.
    pub fn clear_cache(&mut self) {
        self.frame_loader.clear_cache();
        if let Some(ref mut depth_loader) = self.depth_loader {
            depth_loader.clear_cache();
        }
        *self.telemetry_cache.write() = None;
    }

    fn __repr__(&self) -> String {
        format!(
            "EnfusionSession('{}', frames={}, samples={})",
            self.session_id,
            self.num_frames(),
            self.num_samples()
        )
    }

    fn __len__(&self) -> usize {
        self.num_samples()
    }
}

impl EnfusionSession {
    /// Create a session from a directory path.
    pub fn from_path(
        path: &Path,
        config: DatasetConfig,
        telemetry_config: TelemetryConfig,
    ) -> DataResult<Self> {
        if !path.exists() {
            return Err(DataError::Session(format!(
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
        let (frame_paths, enfcap_reader, has_enfcap) = Self::detect_frames(path)?;
        let depth_paths = Self::find_depth_files(path);
        let has_depth = !depth_paths.is_empty() && config.load_depth;
        let has_telemetry = Self::find_telemetry_file(path).is_some();

        // Get frame count and dimensions
        let (num_frames, width, height, fps) = if let Some(ref reader) = enfcap_reader {
            let dims = reader.dimensions();
            (reader.num_frames(), dims.0, dims.1, reader.fps())
        } else {
            // Probe first image for dimensions
            let (w, h) = if !frame_paths.is_empty() {
                Self::probe_image_dimensions(&frame_paths[0])?
            } else {
                return Err(DataError::Session("No frames found".to_string()));
            };
            (frame_paths.len(), w as usize, h as usize, 30.0_f32) // Assume 30fps for images
        };

        let duration_seconds = num_frames as f64 / fps as f64;

        let metadata = SessionMetadata {
            session_id: session_id.clone(),
            num_frames,
            duration_seconds,
            fps,
            width: width as u32,
            height: height as u32,
            has_telemetry,
            has_depth,
            path: path.to_string_lossy().to_string(),
        };

        let frame_loader = FrameLoader::new(Some(config.clone()), true, 1024);
        let depth_loader = if has_depth {
            Some(DepthLoader::new(Some(config.clone()), true, 512))
        } else {
            None
        };

        let telemetry_parser = TelemetryParser::new();

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

    /// Detect frame files in a session directory.
    fn detect_frames(
        path: &Path,
    ) -> DataResult<(Vec<PathBuf>, Option<ENFCAPReader>, bool)> {
        // Check for ENFCAP files first
        let enfcap_path = path.join("video.enfcap");
        if enfcap_path.exists() {
            let reader = ENFCAPReader::open(&enfcap_path)
                .map_err(|e| DataError::Session(format!("Failed to open ENFCAP: {}", e)))?;
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
            .map_err(|e| DataError::Session(format!("Cannot read frames dir: {}", e)))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .map(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp"))
                    .unwrap_or(false)
            })
            .collect();

        frame_paths.sort();

        if frame_paths.is_empty() {
            return Err(DataError::Session(format!(
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
                            .map(|e| e.to_lowercase() == "png")
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
        // Common telemetry file names
        let candidates = [
            "telemetry.csv",
            "controls.csv",
            "data.csv",
            "log.csv",
            "telemetry.json",
            "rlog.json",
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
    fn probe_image_dimensions(path: &Path) -> DataResult<(u32, u32)> {
        let reader = image::io::Reader::open(path)?;
        let dims = reader.into_dimensions()?;
        Ok(dims)
    }

    /// Calculate frame indices for a sample.
    fn sample_frame_indices(&self, sample_idx: usize) -> DataResult<Vec<usize>> {
        let total_frames = self.config.total_frames();
        let stride = self.config.frame_stride;
        let start = sample_idx * stride;

        if start + total_frames * stride > self.metadata.num_frames {
            return Err(DataError::IndexOutOfBounds {
                index: sample_idx,
                length: self.num_samples(),
            });
        }

        let indices: Vec<usize> = (0..total_frames).map(|i| start + i * stride).collect();
        Ok(indices)
    }

    /// Load frames for a sample.
    pub fn load_sample_frames(&mut self, sample_idx: usize) -> DataResult<Array4<f32>> {
        let indices = self.sample_frame_indices(sample_idx)?;

        if self.enfcap_reader.is_some() {
            // TODO: Implement ENFCAP frame reading
            return Err(DataError::Frame("ENFCAP reading not yet implemented".to_string()));
        }

        let paths: Vec<String> = indices
            .iter()
            .map(|&i| self.frame_paths[i].to_string_lossy().to_string())
            .collect();
        self.frame_loader.load_frames_inner(&paths)
    }

    /// Load controls for a sample.
    pub fn load_sample_controls(&mut self, sample_idx: usize) -> DataResult<Array2<f32>> {
        let telemetry = self.load_telemetry()?;
        let indices = self.sample_frame_indices(sample_idx)?;

        let total_frames = indices.len();
        let mut controls = Array2::zeros((total_frames, self.config.control_dim));

        for (i, &frame_idx) in indices.iter().enumerate() {
            if frame_idx < telemetry.nrows() {
                for j in 0..self.config.control_dim.min(telemetry.ncols()) {
                    controls[[i, j]] = telemetry[[frame_idx, j]];
                }
            }
        }

        Ok(controls)
    }

    /// Load depth data for a sample.
    pub fn load_sample_depth(&mut self, sample_idx: usize) -> DataResult<Array4<f32>> {
        let depth_loader = self
            .depth_loader
            .as_ref()
            .ok_or_else(|| DataError::Depth("Depth loader not available".to_string()))?;

        let indices = self.sample_frame_indices(sample_idx)?;

        let paths: Vec<String> = indices
            .iter()
            .filter_map(|&i| self.depth_paths.get(i))
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        if paths.len() != indices.len() {
            return Err(DataError::Depth("Missing depth files".to_string()));
        }

        depth_loader.load_depths_inner(&paths)
    }

    /// Load or retrieve cached telemetry data.
    fn load_telemetry(&mut self) -> DataResult<Array2<f32>> {
        // Check cache
        {
            let cache = self.telemetry_cache.read();
            if let Some(ref telemetry) = *cache {
                return Ok(telemetry.clone());
            }
        }

        // Load telemetry
        let telemetry_path = Self::find_telemetry_file(&self.path)
            .ok_or_else(|| DataError::Telemetry("No telemetry file found".to_string()))?;

        // Parse telemetry file
        let telemetry = self.telemetry_parser.parse_file(&telemetry_path)
            .map_err(|e| DataError::Telemetry(format!("Failed to parse telemetry: {}", e)))?;

        // Convert records to control array [steering, throttle, brake, speed]
        let control_dim = self.config.control_dim.min(4);
        let num_records = telemetry.len();
        let mut controls = Array2::zeros((num_records, control_dim));
        for (i, record) in telemetry.iter().enumerate() {
            if control_dim > 0 { controls[[i, 0]] = record.steering; }
            if control_dim > 1 { controls[[i, 1]] = record.throttle; }
            if control_dim > 2 { controls[[i, 2]] = record.brake; }
            if control_dim > 3 { controls[[i, 3]] = record.speed_ms; }
        }

        // Cache
        {
            let mut cache = self.telemetry_cache.write();
            *cache = Some(controls.clone());
        }

        Ok(controls)
    }
}

/// Find all session directories in a root path.
pub fn find_sessions(root: &Path) -> DataResult<Vec<PathBuf>> {
    let mut sessions = Vec::new();

    if !root.exists() {
        return Err(DataError::Session(format!(
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
                for subentry in fs::read_dir(&path)? {
                    let subentry = subentry?;
                    let subpath = subentry.path();
                    if subpath.is_dir() && is_session_dir(&subpath) {
                        sessions.push(subpath);
                    }
                }
            }
        }
    }

    sessions.sort();
    Ok(sessions)
}

/// Check if a directory looks like a session directory.
fn is_session_dir(path: &Path) -> bool {
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
                    .map(|ext| matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_frame_indices() {
        let config = DatasetConfig {
            sequence_length: 8,
            context_frames: 2,
            frame_stride: 2,
            ..DatasetConfig::default()
        };

        // Total frames needed = (8 + 2) * 2 = 20
        // With 100 frames, samples = (100 - 20) / 2 + 1 = 41
    }

    #[test]
    fn test_is_session_dir() {
        // Would need temp directory setup for proper testing
    }
}
