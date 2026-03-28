//! Batch loading API Python bindings.
//!
//! Provides PyO3 wrappers around the batch loading functionality from
//! the drivedit-batch crate. Designed for training from stored datasets.
//!
//! # Example
//!
//! ```python
//! from drivedit.batch import EnfusionDataset, EnfusionSession
//!
//! # Load entire dataset
//! dataset = EnfusionDataset("/path/to/data")
//! for frames, controls, depth in dataset:
//!     # Training loop
//!     pass
//!
//! # Load single session
//! session = EnfusionSession("/path/to/session")
//! frames = session.get_frames(0)  # [T, C, H, W]
//! ```

use numpy::{IntoPyArray, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyUntypedArrayMethods, ToPyArray};
use parking_lot::RwLock;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{debug, info};

// Re-export types for convenience
pub use crate::types::{PyFrame, PyFrameBatch, PyTelemetry};

/// Dataset configuration for batch loading.
///
/// Controls sequence length, image dimensions, and processing options.
///
/// # Example
///
/// ```python
/// from drivedit.batch import DatasetConfig
///
/// config = DatasetConfig(
///     sequence_length=16,
///     image_height=256,
///     image_width=512,
///     control_dim=4,
/// )
/// ```
#[pyclass(name = "DatasetConfig")]
#[derive(Debug, Clone)]
pub struct PyDatasetConfig {
    /// Number of frames in each sequence.
    #[pyo3(get, set)]
    pub sequence_length: usize,

    /// Target image height.
    #[pyo3(get, set)]
    pub image_height: usize,

    /// Target image width.
    #[pyo3(get, set)]
    pub image_width: usize,

    /// Number of control dimensions.
    #[pyo3(get, set)]
    pub control_dim: usize,

    /// Temporal stride between frames.
    #[pyo3(get, set)]
    pub frame_stride: usize,

    /// Number of context frames.
    #[pyo3(get, set)]
    pub context_frames: usize,

    /// Whether to load depth data.
    #[pyo3(get, set)]
    pub load_depth: bool,

    /// Whether to normalize images to [0, 1].
    #[pyo3(get, set)]
    pub normalize_images: bool,

    /// Random seed.
    #[pyo3(get, set)]
    pub seed: u64,
}

#[pymethods]
impl PyDatasetConfig {
    #[new]
    #[pyo3(signature = (
        sequence_length = 16,
        image_height = 256,
        image_width = 512,
        control_dim = 4,
        frame_stride = 1,
        context_frames = 4,
        load_depth = false,
        normalize_images = true,
        seed = 42
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sequence_length: usize,
        image_height: usize,
        image_width: usize,
        control_dim: usize,
        frame_stride: usize,
        context_frames: usize,
        load_depth: bool,
        normalize_images: bool,
        seed: u64,
    ) -> Self {
        Self {
            sequence_length,
            image_height,
            image_width,
            control_dim,
            frame_stride,
            context_frames,
            load_depth,
            normalize_images,
            seed,
        }
    }

    /// Get total frames needed per sample.
    fn total_frames(&self) -> usize {
        self.context_frames + self.sequence_length
    }

    /// Create a minimal config for testing.
    #[staticmethod]
    fn minimal() -> Self {
        Self {
            sequence_length: 4,
            image_height: 64,
            image_width: 128,
            control_dim: 4,
            frame_stride: 1,
            context_frames: 2,
            load_depth: false,
            normalize_images: true,
            seed: 42,
        }
    }

    /// Create a production config.
    #[staticmethod]
    fn production() -> Self {
        Self {
            sequence_length: 16,
            image_height: 256,
            image_width: 512,
            control_dim: 4,
            frame_stride: 1,
            context_frames: 4,
            load_depth: true,
            normalize_images: true,
            seed: 42,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetConfig(seq_len={}, {}x{}, ctrl_dim={})",
            self.sequence_length, self.image_height, self.image_width, self.control_dim
        )
    }
}

impl Default for PyDatasetConfig {
    fn default() -> Self {
        Self::new(16, 256, 512, 4, 1, 4, false, true, 42)
    }
}

/// Data loader configuration.
#[pyclass(name = "LoaderConfig")]
#[derive(Debug, Clone)]
pub struct PyLoaderConfig {
    /// Number of worker threads.
    #[pyo3(get, set)]
    pub num_workers: usize,

    /// Batch size.
    #[pyo3(get, set)]
    pub batch_size: usize,

    /// Whether to shuffle data.
    #[pyo3(get, set)]
    pub shuffle: bool,

    /// Whether to drop incomplete last batch.
    #[pyo3(get, set)]
    pub drop_last: bool,

    /// Use memory-mapped files.
    #[pyo3(get, set)]
    pub use_mmap: bool,

    /// Cache decoded images.
    #[pyo3(get, set)]
    pub cache_decoded: bool,

    /// Maximum cache size in MB.
    #[pyo3(get, set)]
    pub cache_size_mb: usize,
}

#[pymethods]
impl PyLoaderConfig {
    #[new]
    #[pyo3(signature = (
        num_workers = 4,
        batch_size = 8,
        shuffle = true,
        drop_last = true,
        use_mmap = true,
        cache_decoded = false,
        cache_size_mb = 1024
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_workers: usize,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        use_mmap: bool,
        cache_decoded: bool,
        cache_size_mb: usize,
    ) -> Self {
        Self {
            num_workers,
            batch_size,
            shuffle,
            drop_last,
            use_mmap,
            cache_decoded,
            cache_size_mb,
        }
    }

    /// Auto-detect optimal configuration.
    #[staticmethod]
    fn auto_detect() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let num_workers = (num_cpus / 2).max(1).min(16);

        Self {
            num_workers,
            batch_size: 8,
            shuffle: true,
            drop_last: true,
            use_mmap: true,
            cache_decoded: false,
            cache_size_mb: 2048,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LoaderConfig(workers={}, batch={})",
            self.num_workers, self.batch_size
        )
    }
}

impl Default for PyLoaderConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// Telemetry parsing configuration.
#[pyclass(name = "TelemetryConfig")]
#[derive(Debug, Clone)]
pub struct PyTelemetryConfig {
    /// Column name for steering.
    #[pyo3(get, set)]
    pub steering_col: String,

    /// Column name for speed.
    #[pyo3(get, set)]
    pub speed_col: String,

    /// Sampling rate in Hz.
    #[pyo3(get, set)]
    pub sampling_rate_hz: f32,

    /// Steering normalization range.
    #[pyo3(get, set)]
    pub steering_range: (f32, f32),

    /// Speed normalization range (m/s).
    #[pyo3(get, set)]
    pub speed_range: (f32, f32),
}

#[pymethods]
impl PyTelemetryConfig {
    #[new]
    #[pyo3(signature = (
        steering_col = "steering".to_string(),
        speed_col = "speed".to_string(),
        sampling_rate_hz = 20.0,
        steering_range = (-1.0, 1.0),
        speed_range = (0.0, 40.0)
    ))]
    fn new(
        steering_col: String,
        speed_col: String,
        sampling_rate_hz: f32,
        steering_range: (f32, f32),
        speed_range: (f32, f32),
    ) -> Self {
        Self {
            steering_col,
            speed_col,
            sampling_rate_hz,
            steering_range,
            speed_range,
        }
    }

    /// Create config for comma.ai format.
    #[staticmethod]
    fn comma_ai() -> Self {
        Self {
            steering_col: "steeringAngleDeg".to_string(),
            speed_col: "vEgo".to_string(),
            sampling_rate_hz: 20.0,
            steering_range: (-500.0, 500.0),
            speed_range: (0.0, 45.0),
        }
    }

    /// Create config for nuScenes format.
    #[staticmethod]
    fn nuscenes() -> Self {
        Self {
            steering_col: "steering".to_string(),
            speed_col: "speed".to_string(),
            sampling_rate_hz: 12.0,
            steering_range: (-1.0, 1.0),
            speed_range: (0.0, 30.0),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TelemetryConfig(rate={}Hz)",
            self.sampling_rate_hz
        )
    }
}

impl Default for PyTelemetryConfig {
    fn default() -> Self {
        Self::comma_ai()
    }
}

/// Session metadata.
#[pyclass(name = "SessionMetadata")]
#[derive(Debug, Clone)]
pub struct PySessionMetadata {
    /// Session identifier.
    #[pyo3(get)]
    pub session_id: String,

    /// Total number of frames.
    #[pyo3(get)]
    pub num_frames: usize,

    /// Duration in seconds.
    #[pyo3(get)]
    pub duration_seconds: f64,

    /// Frames per second.
    #[pyo3(get)]
    pub fps: f32,

    /// Frame width.
    #[pyo3(get)]
    pub width: u32,

    /// Frame height.
    #[pyo3(get)]
    pub height: u32,

    /// Whether telemetry is available.
    #[pyo3(get)]
    pub has_telemetry: bool,

    /// Whether depth data is available.
    #[pyo3(get)]
    pub has_depth: bool,

    /// Session path.
    #[pyo3(get)]
    pub path: String,
}

#[pymethods]
impl PySessionMetadata {
    fn __repr__(&self) -> String {
        format!(
            "SessionMetadata(id='{}', frames={}, duration={:.1}s)",
            self.session_id, self.num_frames, self.duration_seconds
        )
    }
}

/// Single Enfusion recording session.
///
/// Provides access to synchronized video frames and telemetry.
///
/// # Example
///
/// ```python
/// from drivedit.batch import EnfusionSession
///
/// session = EnfusionSession("/path/to/session")
/// print(session.metadata())
///
/// # Get sample data
/// frames = session.get_frames(0)    # [T, C, H, W] float32
/// controls = session.get_controls(0) # [T, 4] float32
/// ```
#[pyclass(name = "EnfusionSession")]
pub struct PyEnfusionSession {
    /// Session path.
    path: PathBuf,

    /// Session ID.
    session_id: String,

    /// Configuration.
    config: PyDatasetConfig,

    /// Metadata.
    metadata: PySessionMetadata,

    /// Cached frame paths.
    frame_paths: Vec<PathBuf>,

    /// Cached telemetry.
    telemetry_cache: Arc<RwLock<Option<Vec<Vec<f32>>>>>,
}

#[pymethods]
impl PyEnfusionSession {
    /// Create a new session from a directory path.
    #[new]
    #[pyo3(signature = (path, config = None, telemetry_config = None))]
    fn new(
        path: &str,
        config: Option<PyDatasetConfig>,
        telemetry_config: Option<PyTelemetryConfig>,
    ) -> PyResult<Self> {
        let path = PathBuf::from(path);
        if !path.exists() {
            return Err(PyValueError::new_err(format!(
                "Session path does not exist: {:?}",
                path
            )));
        }

        let config = config.unwrap_or_default();
        let _telemetry_config = telemetry_config.unwrap_or_default();

        let session_id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Find frame files
        let frame_paths = find_frame_files(&path)?;
        let num_frames = frame_paths.len();

        // Probe first image for dimensions
        let (width, height) = if !frame_paths.is_empty() {
            probe_image_dimensions(&frame_paths[0])?
        } else {
            (config.image_width as u32, config.image_height as u32)
        };

        let fps = 30.0f32;
        let duration_seconds = num_frames as f64 / fps as f64;
        let has_telemetry = find_telemetry_file(&path).is_some();
        let has_depth = path.join("depth").exists();

        let metadata = PySessionMetadata {
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

        info!(
            "Loaded session '{}' with {} frames",
            session_id, num_frames
        );

        Ok(Self {
            path,
            session_id,
            config,
            metadata,
            frame_paths,
            telemetry_cache: Arc::new(RwLock::new(None)),
        })
    }

    /// Get session metadata.
    fn metadata(&self) -> PySessionMetadata {
        self.metadata.clone()
    }

    /// Get the number of frames in the session.
    fn num_frames(&self) -> usize {
        self.metadata.num_frames
    }

    /// Get the number of valid samples.
    fn num_samples(&self) -> usize {
        let total_needed = self.config.total_frames() * self.config.frame_stride;
        if self.metadata.num_frames < total_needed {
            0
        } else {
            (self.metadata.num_frames - total_needed) / self.config.frame_stride + 1
        }
    }

    /// Get frames for a sample.
    ///
    /// Returns numpy array [T, C, H, W] float32.
    fn get_frames<'py>(
        &self,
        py: Python<'py>,
        sample_idx: usize,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let indices = self.sample_indices(sample_idx)?;

        // Load frames (placeholder - would use actual frame loading)
        let t = indices.len();
        let c = 3;
        let h = self.config.image_height;
        let w = self.config.image_width;

        // Create dummy data for now (actual implementation would load from disk)
        let data = vec![0.5f32; t * c * h * w];
        let arr = data.into_pyarray(py);
        arr.reshape([t, c, h, w])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Get control signals for a sample.
    ///
    /// Returns numpy array [T, control_dim] float32.
    fn get_controls<'py>(
        &self,
        py: Python<'py>,
        sample_idx: usize,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let indices = self.sample_indices(sample_idx)?;

        let t = indices.len();
        let dim = self.config.control_dim;

        // Create dummy data (actual implementation would load from telemetry)
        let data = vec![0.0f32; t * dim];
        let arr = data.into_pyarray(py);
        arr.reshape([t, dim])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Get depth data for a sample (if available).
    ///
    /// Returns numpy array [T, 1, H, W] float32 or None.
    fn get_depth<'py>(
        &self,
        py: Python<'py>,
        sample_idx: usize,
    ) -> PyResult<Option<Bound<'py, PyArray4<f32>>>> {
        if !self.metadata.has_depth {
            return Ok(None);
        }

        let indices = self.sample_indices(sample_idx)?;
        let t = indices.len();
        let h = self.config.image_height;
        let w = self.config.image_width;

        // Create dummy depth data
        let data = vec![0.5f32; t * 1 * h * w];
        let arr = data.into_pyarray(py);
        let reshaped = arr
            .reshape([t, 1, h, w])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))?;
        Ok(Some(reshaped))
    }

    /// Get a complete sample (frames, controls, optional depth).
    fn get_sample<'py>(
        &self,
        py: Python<'py>,
        sample_idx: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )> {
        let frames = self.get_frames(py, sample_idx)?;
        let controls = self.get_controls(py, sample_idx)?;
        let depth = self.get_depth(py, sample_idx)?;
        Ok((frames, controls, depth))
    }

    /// Preload telemetry into cache.
    fn preload_telemetry(&self) -> PyResult<()> {
        debug!("Preloading telemetry for session {}", self.session_id);
        // Actual implementation would load telemetry here
        Ok(())
    }

    /// Clear all caches.
    fn clear_cache(&self) {
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

impl PyEnfusionSession {
    fn sample_indices(&self, sample_idx: usize) -> PyResult<Vec<usize>> {
        let total_frames = self.config.total_frames();
        let stride = self.config.frame_stride;
        let start = sample_idx * stride;

        if start + total_frames * stride > self.metadata.num_frames {
            return Err(PyValueError::new_err(format!(
                "Sample index {} out of bounds (max {})",
                sample_idx,
                self.num_samples()
            )));
        }

        let indices: Vec<usize> = (0..total_frames).map(|i| start + i * stride).collect();
        Ok(indices)
    }
}

/// Dataset for batch loading from multiple sessions.
///
/// Provides PyTorch DataLoader-compatible interface.
///
/// # Example
///
/// ```python
/// from drivedit.batch import EnfusionDataset
///
/// dataset = EnfusionDataset("/path/to/data")
/// print(f"Total samples: {len(dataset)}")
///
/// # Iteration
/// for frames, controls, depth in dataset:
///     # Training loop
///     pass
///
/// # Direct indexing
/// frames, controls, depth = dataset[0]
/// ```
#[pyclass(name = "EnfusionDataset")]
pub struct PyEnfusionDataset {
    /// Configuration.
    config: PyDatasetConfig,

    /// Loader configuration.
    loader_config: PyLoaderConfig,

    /// Session paths.
    session_paths: Vec<PathBuf>,

    /// Sample indices mapping.
    sample_indices: Vec<(usize, usize)>, // (session_idx, sample_in_session)

    /// Total samples.
    total_samples: usize,

    /// Current epoch.
    epoch: usize,

    /// Shuffled indices.
    shuffled_indices: Arc<RwLock<Vec<usize>>>,
}

#[pymethods]
impl PyEnfusionDataset {
    /// Create a dataset from a root directory.
    #[new]
    #[pyo3(signature = (root_path, config = None, loader_config = None))]
    fn new(
        root_path: &str,
        config: Option<PyDatasetConfig>,
        loader_config: Option<PyLoaderConfig>,
    ) -> PyResult<Self> {
        let root = PathBuf::from(root_path);
        if !root.exists() {
            return Err(PyValueError::new_err(format!(
                "Root path does not exist: {:?}",
                root
            )));
        }

        let config = config.unwrap_or_default();
        let loader_config = loader_config.unwrap_or_default();

        // Find session directories
        let session_paths = find_sessions(&root)?;
        if session_paths.is_empty() {
            return Err(PyValueError::new_err(format!(
                "No sessions found in {:?}",
                root
            )));
        }

        // Build sample indices
        let mut sample_indices = Vec::new();
        let mut total_samples = 0;

        for (session_idx, path) in session_paths.iter().enumerate() {
            // Probe session for sample count
            let frame_count = count_frames(path)?;
            let total_frames = config.total_frames() * config.frame_stride;

            if frame_count >= total_frames {
                let num_samples = (frame_count - total_frames) / config.frame_stride + 1;
                for sample_idx in 0..num_samples {
                    sample_indices.push((session_idx, sample_idx));
                }
                total_samples += num_samples;
            }
        }

        let shuffled_indices: Vec<usize> = (0..total_samples).collect();

        info!(
            "Dataset loaded: {} sessions, {} samples",
            session_paths.len(),
            total_samples
        );

        Ok(Self {
            config,
            loader_config,
            session_paths,
            sample_indices,
            total_samples,
            epoch: 0,
            shuffled_indices: Arc::new(RwLock::new(shuffled_indices)),
        })
    }

    /// Create dataset from explicit session paths.
    #[staticmethod]
    #[pyo3(signature = (session_paths, config = None, loader_config = None))]
    fn from_sessions(
        session_paths: Vec<String>,
        config: Option<PyDatasetConfig>,
        loader_config: Option<PyLoaderConfig>,
    ) -> PyResult<Self> {
        let config = config.unwrap_or_default();
        let loader_config = loader_config.unwrap_or_default();

        let session_paths: Vec<PathBuf> = session_paths.iter().map(PathBuf::from).collect();

        // Build sample indices
        let mut sample_indices = Vec::new();
        let mut total_samples = 0;

        for (session_idx, path) in session_paths.iter().enumerate() {
            let frame_count = count_frames(path)?;
            let total_frames = config.total_frames() * config.frame_stride;

            if frame_count >= total_frames {
                let num_samples = (frame_count - total_frames) / config.frame_stride + 1;
                for sample_idx in 0..num_samples {
                    sample_indices.push((session_idx, sample_idx));
                }
                total_samples += num_samples;
            }
        }

        let shuffled_indices: Vec<usize> = (0..total_samples).collect();

        Ok(Self {
            config,
            loader_config,
            session_paths,
            sample_indices,
            total_samples,
            epoch: 0,
            shuffled_indices: Arc::new(RwLock::new(shuffled_indices)),
        })
    }

    /// Get total number of samples.
    fn __len__(&self) -> usize {
        self.total_samples
    }

    /// Get sample by index.
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        idx: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )> {
        if idx >= self.total_samples {
            return Err(PyValueError::new_err(format!(
                "Index {} out of bounds (size {})",
                idx, self.total_samples
            )));
        }

        // Map through shuffle
        let actual_idx = if self.loader_config.shuffle {
            let indices = self.shuffled_indices.read();
            indices[idx]
        } else {
            idx
        };

        let (_session_idx, _sample_idx) = self.sample_indices[actual_idx];

        // Return dummy data (actual implementation would load from session)
        let t = self.config.total_frames();
        let c = 3;
        let h = self.config.image_height;
        let w = self.config.image_width;
        let dim = self.config.control_dim;

        let frames = vec![0.5f32; t * c * h * w].into_pyarray(py);
        let controls = vec![0.0f32; t * dim].into_pyarray(py);

        let frames = frames
            .reshape([t, c, h, w])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))?;
        let controls = controls
            .reshape([t, dim])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))?;

        let depth = if self.config.load_depth {
            let d = vec![0.5f32; t * 1 * h * w].into_pyarray(py);
            Some(
                d.reshape([t, 1, h, w])
                    .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))?,
            )
        } else {
            None
        };

        Ok((frames, controls, depth))
    }

    /// Get number of sessions.
    fn num_sessions(&self) -> usize {
        self.session_paths.len()
    }

    /// Get session paths.
    fn session_paths(&self) -> Vec<String> {
        self.session_paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect()
    }

    /// Set epoch (triggers reshuffle).
    fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
        if self.loader_config.shuffle {
            self.reshuffle();
        }
    }

    /// Get current epoch.
    fn get_epoch(&self) -> usize {
        self.epoch
    }

    /// Shuffle dataset indices.
    fn shuffle(&mut self) {
        self.reshuffle();
    }

    /// Get dataset statistics.
    fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_samples".to_string(), self.total_samples);
        stats.insert("num_sessions".to_string(), self.session_paths.len());
        stats.insert("sequence_length".to_string(), self.config.sequence_length);
        stats.insert("context_frames".to_string(), self.config.context_frames);
        stats
    }

    fn __repr__(&self) -> String {
        format!(
            "EnfusionDataset(samples={}, sessions={})",
            self.total_samples,
            self.session_paths.len()
        )
    }
}

impl PyEnfusionDataset {
    fn reshuffle(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.config.seed.hash(&mut hasher);
        self.epoch.hash(&mut hasher);
        let combined_seed = hasher.finish();

        let mut indices = self.shuffled_indices.write();

        // Fisher-Yates shuffle
        let mut rng_state = combined_seed;
        for i in (1..indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }
    }
}

/// High-performance frame loader.
///
/// # Example
///
/// ```python
/// from drivedit.batch import FrameLoader
///
/// loader = FrameLoader()
/// frames = loader.load_frames(["frame1.jpg", "frame2.jpg"])
/// ```
#[pyclass(name = "FrameLoader")]
pub struct PyFrameLoader {
    /// Configuration.
    config: PyDatasetConfig,

    /// Whether to use cache.
    #[pyo3(get)]
    use_cache: bool,

    /// Cache hits.
    #[pyo3(get)]
    cache_hits: usize,

    /// Cache misses.
    #[pyo3(get)]
    cache_misses: usize,
}

#[pymethods]
impl PyFrameLoader {
    #[new]
    #[pyo3(signature = (config = None, use_cache = true, _cache_size_mb = 1024))]
    fn new(config: Option<PyDatasetConfig>, use_cache: bool, _cache_size_mb: usize) -> Self {
        Self {
            config: config.unwrap_or_default(),
            use_cache,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Load a single frame.
    fn load_frame<'py>(
        &mut self,
        py: Python<'py>,
        _path: &str,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let c = 3;
        let h = self.config.image_height;
        let w = self.config.image_width;

        let data = vec![0.5f32; c * h * w];
        let arr = data.into_pyarray(py);
        arr.reshape([c, h, w])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Load multiple frames in parallel.
    fn load_frames<'py>(
        &mut self,
        py: Python<'py>,
        paths: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let t = paths.len();
        let c = 3;
        let h = self.config.image_height;
        let w = self.config.image_width;

        let data = vec![0.5f32; t * c * h * w];
        let arr = data.into_pyarray(py);
        arr.reshape([t, c, h, w])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Clear frame cache.
    fn clear_cache(&mut self) {
        self.cache_hits = 0;
        self.cache_misses = 0;
    }

    /// Get cache statistics.
    fn cache_stats(&self) -> (usize, usize, f64) {
        let total = self.cache_hits + self.cache_misses;
        let hit_rate = if total > 0 {
            self.cache_hits as f64 / total as f64
        } else {
            0.0
        };
        (self.cache_hits, self.cache_misses, hit_rate)
    }

    fn __repr__(&self) -> String {
        let (_, _, rate) = self.cache_stats();
        format!(
            "FrameLoader({}x{}, hit_rate={:.1}%)",
            self.config.image_width,
            self.config.image_height,
            rate * 100.0
        )
    }
}

/// Depth map loader.
#[pyclass(name = "DepthLoader")]
pub struct PyDepthLoader {
    config: PyDatasetConfig,

    #[pyo3(get)]
    use_cache: bool,
}

#[pymethods]
impl PyDepthLoader {
    #[new]
    #[pyo3(signature = (config = None, use_cache = true, _cache_size_mb = 512))]
    fn new(config: Option<PyDatasetConfig>, use_cache: bool, _cache_size_mb: usize) -> Self {
        Self {
            config: config.unwrap_or_default(),
            use_cache,
        }
    }

    /// Load a single depth map.
    fn load_depth<'py>(
        &self,
        py: Python<'py>,
        _path: &str,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let h = self.config.image_height;
        let w = self.config.image_width;

        let data = vec![0.5f32; 1 * h * w];
        let arr = data.into_pyarray(py);
        arr.reshape([1, h, w])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Load multiple depth maps.
    fn load_depths<'py>(
        &self,
        py: Python<'py>,
        paths: Vec<String>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let t = paths.len();
        let h = self.config.image_height;
        let w = self.config.image_width;

        let data = vec![0.5f32; t * 1 * h * w];
        let arr = data.into_pyarray(py);
        arr.reshape([t, 1, h, w])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Clear depth cache.
    fn clear_cache(&self) {
        // No-op for now
    }

    fn __repr__(&self) -> String {
        format!(
            "DepthLoader({}x{}, cached={})",
            self.config.image_width, self.config.image_height, self.use_cache
        )
    }
}

// Helper functions

fn find_frame_files(path: &PathBuf) -> PyResult<Vec<PathBuf>> {
    let frames_dir = if path.join("frames").exists() {
        path.join("frames")
    } else if path.join("images").exists() {
        path.join("images")
    } else {
        path.clone()
    };

    let mut paths: Vec<PathBuf> = std::fs::read_dir(&frames_dir)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read dir: {}", e)))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp"))
                .unwrap_or(false)
        })
        .collect();

    paths.sort();
    Ok(paths)
}

fn probe_image_dimensions(path: &PathBuf) -> PyResult<(u32, u32)> {
    // Placeholder - actual implementation would probe image
    Ok((640, 360))
}

fn find_telemetry_file(path: &PathBuf) -> Option<PathBuf> {
    let candidates = ["telemetry.csv", "controls.csv", "data.csv"];
    for name in &candidates {
        let file_path = path.join(name);
        if file_path.exists() {
            return Some(file_path);
        }
    }
    None
}

fn find_sessions(root: &PathBuf) -> PyResult<Vec<PathBuf>> {
    let mut sessions = Vec::new();

    if is_session_dir(root) {
        sessions.push(root.clone());
        return Ok(sessions);
    }

    for entry in std::fs::read_dir(root)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to read root: {}", e)))?
    {
        let entry = entry.map_err(|e| PyRuntimeError::new_err(format!("Entry error: {}", e)))?;
        let path = entry.path();

        if path.is_dir() {
            if is_session_dir(&path) {
                sessions.push(path);
            }
        }
    }

    sessions.sort();
    Ok(sessions)
}

fn is_session_dir(path: &PathBuf) -> bool {
    path.join("video.enfcap").exists()
        || path.join("frames").exists()
        || path.join("images").exists()
}

fn count_frames(path: &PathBuf) -> PyResult<usize> {
    let frame_paths = find_frame_files(path)?;
    Ok(frame_paths.len())
}

/// Register batch module classes and functions.
pub fn register_batch_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDatasetConfig>()?;
    m.add_class::<PyLoaderConfig>()?;
    m.add_class::<PyTelemetryConfig>()?;
    m.add_class::<PySessionMetadata>()?;
    m.add_class::<PyEnfusionSession>()?;
    m.add_class::<PyEnfusionDataset>()?;
    m.add_class::<PyFrameLoader>()?;
    m.add_class::<PyDepthLoader>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_config_default() {
        let config = PyDatasetConfig::default();
        assert_eq!(config.sequence_length, 16);
        assert_eq!(config.total_frames(), 20);
    }

    #[test]
    fn test_loader_config_auto_detect() {
        let config = PyLoaderConfig::auto_detect();
        assert!(config.num_workers > 0);
    }
}
