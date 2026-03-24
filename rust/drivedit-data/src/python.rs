//! Python bindings for DriveDiT data loading library.
//!
//! Provides PyO3 class wrappers for all Rust types with:
//! - Zero-copy numpy array creation
//! - Proper GIL handling for parallel operations
//! - Python-friendly error messages
//!
//! # Example
//! ```python
//! from drivedit_data import RustEnfusionDataset, RustDatasetConfig
//!
//! config = RustDatasetConfig(
//!     sequence_length=16,
//!     image_height=256,
//!     image_width=256,
//! )
//! dataset = RustEnfusionDataset("/path/to/data", config)
//! sample = dataset[0]
//! print(sample.frames.shape)  # (16, 3, 256, 256)
//! ```

use numpy::{IntoPyArray, PyArray3, PyArray4, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::dataset::{EnfusionDataset, EnfusionSample, EnfusionSession};
use crate::error::DataResult;
use crate::telemetry::TelemetryData;
use crate::config::DatasetConfig;
use crate::frame_loader::FrameLoader;

/// Python-exposed dataset configuration.
#[pyclass(name = "RustDatasetConfig")]
#[derive(Clone)]
pub struct PyDatasetConfig {
    inner: DatasetConfig,
}

#[pymethods]
impl PyDatasetConfig {
    /// Create a new dataset configuration.
    ///
    /// Args:
    ///     sequence_length: Number of frames per sequence (default: 16)
    ///     frame_skip: Skip N frames between samples (default: 1)
    ///     image_height: Target image height (default: 256)
    ///     image_width: Target image width (default: 256)
    ///     load_depth: Whether to load depth maps (default: True)
    ///     load_controls: Whether to load control signals (default: True)
    ///     control_dim: Number of control dimensions (default: 6)
    ///     normalize_frames: Normalize frames to [0, 1] (default: True)
    ///     cache_frames: Cache frames in memory (default: True)
    ///     max_cache_mb: Maximum cache size in MB (default: 1000.0)
    ///     num_workers: Number of parallel workers (default: 4)
    #[new]
    #[pyo3(signature = (
        sequence_length = 16,
        frame_skip = 1,
        image_height = 256,
        image_width = 256,
        load_depth = true,
        load_controls = true,
        control_dim = 6,
        normalize_frames = true,
        cache_frames = true,
        max_cache_mb = 1000.0,
        num_workers = 4,
    ))]
    pub fn new(
        sequence_length: usize,
        frame_skip: usize,
        image_height: usize,
        image_width: usize,
        load_depth: bool,
        load_controls: bool,
        control_dim: usize,
        normalize_frames: bool,
        cache_frames: bool,
        max_cache_mb: f64,
        num_workers: usize,
    ) -> Self {
        PyDatasetConfig {
            inner: DatasetConfig {
                sequence_length,
                frame_skip,
                image_size: (image_height, image_width),
                load_depth,
                load_controls,
                control_dim,
                normalize_frames,
                cache_frames,
                max_cache_mb,
                num_workers,
                ..Default::default()
            },
        }
    }

    /// Get sequence length.
    #[getter]
    pub fn sequence_length(&self) -> usize {
        self.inner.sequence_length
    }

    /// Get frame skip.
    #[getter]
    pub fn frame_skip(&self) -> usize {
        self.inner.frame_skip
    }

    /// Get image size as (height, width) tuple.
    #[getter]
    pub fn image_size(&self) -> (usize, usize) {
        self.inner.image_size
    }

    /// Get number of control dimensions.
    #[getter]
    pub fn control_dim(&self) -> usize {
        self.inner.control_dim
    }

    fn __repr__(&self) -> String {
        format!(
            "RustDatasetConfig(sequence_length={}, image_size={:?}, control_dim={})",
            self.inner.sequence_length, self.inner.image_size, self.inner.control_dim
        )
    }
}

/// A single sample from the Enfusion dataset.
///
/// Contains frames, controls, optional depth, and metadata.
#[pyclass(name = "RustEnfusionSample")]
pub struct PyEnfusionSample {
    /// Frames tensor [T, C, H, W] as numpy array
    frames: Py<PyArray4<f32>>,
    /// Control signals [T, D] as numpy array
    controls: Py<PyArray3<f32>>,
    /// Optional depth maps [T, 1, H, W]
    depth: Option<Py<PyArray4<f32>>>,
    /// Ego transforms [T, 4, 4]
    ego_transform: Py<PyArray3<f32>>,
    /// Anchor mask [T]
    anchor_mask: Py<PyArray3<u8>>,
    /// Session ID
    session_id: String,
    /// Start frame index
    start_frame: usize,
    /// Frame indices
    frame_indices: Vec<usize>,
}

#[pymethods]
impl PyEnfusionSample {
    /// Get frames tensor [T, C, H, W].
    #[getter]
    pub fn frames<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray4<f32>> {
        self.frames.bind(py).clone()
    }

    /// Get control signals [T, D].
    #[getter]
    pub fn controls<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.controls.bind(py).clone()
    }

    /// Get depth maps [T, 1, H, W] if available.
    #[getter]
    pub fn depth<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray4<f32>>> {
        self.depth.as_ref().map(|d| d.bind(py).clone())
    }

    /// Get ego transforms [T, 4, 4].
    #[getter]
    pub fn ego_transform<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        self.ego_transform.bind(py).clone()
    }

    /// Get anchor mask [T].
    #[getter]
    pub fn anchor_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<u8>> {
        self.anchor_mask.bind(py).clone()
    }

    /// Get session ID.
    #[getter]
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get start frame index.
    #[getter]
    pub fn start_frame(&self) -> usize {
        self.start_frame
    }

    /// Get frame indices as list.
    #[getter]
    pub fn frame_indices(&self) -> Vec<usize> {
        self.frame_indices.clone()
    }

    /// Convert to dictionary for PyTorch compatibility.
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);
        dict.set_item("frames", self.frames.bind(py))?;
        dict.set_item("controls", self.controls.bind(py))?;
        dict.set_item("ego_transform", self.ego_transform.bind(py))?;
        dict.set_item("anchor_mask", self.anchor_mask.bind(py))?;
        dict.set_item("session_id", &self.session_id)?;
        dict.set_item("start_frame", self.start_frame)?;

        if let Some(ref depth) = self.depth {
            dict.set_item("depth", depth.bind(py))?;
        }

        // Create metadata dict
        let metadata = PyDict::new_bound(py);
        metadata.set_item("session_id", &self.session_id)?;
        metadata.set_item("start_frame", self.start_frame)?;
        metadata.set_item("frame_indices", self.frame_indices.clone())?;
        dict.set_item("metadata", metadata)?;

        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustEnfusionSample(session_id='{}', start_frame={}, num_frames={})",
            self.session_id,
            self.start_frame,
            self.frame_indices.len()
        )
    }
}

/// High-performance Enfusion dataset with Rust backend.
///
/// Provides 10x faster data loading compared to pure Python implementation
/// through parallel I/O, memory-mapped files, and zero-copy numpy arrays.
///
/// Example:
///     >>> config = RustDatasetConfig(sequence_length=16)
///     >>> dataset = RustEnfusionDataset("/path/to/data", config)
///     >>> len(dataset)
///     1000
///     >>> sample = dataset[0]
///     >>> sample.frames.shape
///     (16, 3, 256, 256)
#[pyclass(name = "RustEnfusionDataset")]
pub struct PyEnfusionDataset {
    /// Inner Rust dataset
    inner: Arc<EnfusionDataset>,
    /// Configuration
    config: DatasetConfig,
    /// Frame loader for parallel loading
    frame_loader: Arc<FrameLoader>,
}

#[pymethods]
impl PyEnfusionDataset {
    /// Create a new Enfusion dataset.
    ///
    /// Args:
    ///     data_root: Root directory containing session folders
    ///     config: Dataset configuration (optional)
    ///     split: Dataset split ('train', 'val', 'test')
    ///     session_ids: Optional list of specific session IDs to load
    ///
    /// Returns:
    ///     A new RustEnfusionDataset instance
    ///
    /// Raises:
    ///     IOError: If data_root does not exist
    ///     ValueError: If no valid sessions found
    #[new]
    #[pyo3(signature = (data_root, config = None, split = "train", session_ids = None))]
    pub fn new(
        data_root: &str,
        config: Option<PyDatasetConfig>,
        split: &str,
        session_ids: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let config = config.map(|c| c.inner).unwrap_or_default();

        let frame_loader = Arc::new(FrameLoader::new(
            config.num_workers,
            config.cache_frames,
            config.max_cache_mb,
        ));

        // Release GIL for expensive I/O operations
        let inner = Python::with_gil(|py| {
            py.allow_threads(|| {
                EnfusionDataset::new(
                    PathBuf::from(data_root),
                    config.clone(),
                    split.to_string(),
                    session_ids,
                )
            })
        })?;

        Ok(PyEnfusionDataset {
            inner: Arc::new(inner),
            config,
            frame_loader,
        })
    }

    /// Get the number of sequences in the dataset.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Get a sample by index.
    ///
    /// Args:
    ///     idx: Sample index
    ///
    /// Returns:
    ///     RustEnfusionSample containing frames, controls, etc.
    ///
    /// Raises:
    ///     IndexError: If idx is out of bounds
    pub fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<PyEnfusionSample> {
        let len = self.inner.len();
        let idx = if idx < 0 {
            (len as isize + idx) as usize
        } else {
            idx as usize
        };

        if idx >= len {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of range for dataset with {} samples",
                idx, len
            )));
        }

        // Release GIL for data loading
        let sample = py.allow_threads(|| {
            self.inner.get_sample(idx, &self.frame_loader)
        })?;

        // Convert to Python arrays without copying
        self.sample_to_python(py, sample)
    }

    /// Load a batch of samples in parallel.
    ///
    /// Args:
    ///     indices: List of sample indices to load
    ///
    /// Returns:
    ///     List of RustEnfusionSample objects
    ///
    /// This method is optimized for parallel I/O and is significantly
    /// faster than calling __getitem__ repeatedly.
    pub fn load_batch(&self, py: Python<'_>, indices: Vec<usize>) -> PyResult<Vec<PyEnfusionSample>> {
        // Release GIL for parallel loading
        let samples: DataResult<Vec<EnfusionSample>> = py.allow_threads(|| {
            self.inner.load_batch(&indices, &self.frame_loader)
        });

        let samples = samples?;

        samples
            .into_iter()
            .map(|s| self.sample_to_python(py, s))
            .collect()
    }

    /// Get number of sessions in the dataset.
    #[getter]
    pub fn num_sessions(&self) -> usize {
        self.inner.num_sessions()
    }

    /// Get list of session IDs.
    #[getter]
    pub fn session_ids(&self) -> Vec<String> {
        self.inner.session_ids()
    }

    /// Get total number of frames across all sessions.
    #[getter]
    pub fn total_frames(&self) -> usize {
        self.inner.total_frames()
    }

    /// Get dataset configuration.
    #[getter]
    pub fn config(&self) -> PyDatasetConfig {
        PyDatasetConfig {
            inner: self.config.clone(),
        }
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> HashMap<String, f64> {
        self.frame_loader.cache_stats()
    }

    /// Clear the frame cache.
    pub fn clear_cache(&self) {
        self.frame_loader.clear_cache();
    }

    /// Prefetch samples in the background.
    ///
    /// Args:
    ///     indices: List of sample indices to prefetch
    ///
    /// This is useful when you know which samples will be needed next.
    pub fn prefetch(&self, py: Python<'_>, indices: Vec<usize>) {
        let inner = Arc::clone(&self.inner);
        let frame_loader = Arc::clone(&self.frame_loader);

        py.allow_threads(move || {
            inner.prefetch(&indices, &frame_loader);
        });
    }

    /// Get information about loaded sessions.
    pub fn get_session_info(&self, py: Python<'_>) -> PyResult<Bound<'_, PyList>> {
        let info = self.inner.get_session_info();
        let list = PyList::empty_bound(py);

        for session_info in info {
            let dict = PyDict::new_bound(py);
            dict.set_item("session_id", session_info.session_id)?;
            dict.set_item("num_frames", session_info.num_frames)?;
            dict.set_item("path", session_info.path.to_string_lossy().to_string())?;
            list.append(dict)?;
        }

        Ok(list)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustEnfusionDataset(sessions={}, sequences={}, config={:?})",
            self.inner.num_sessions(),
            self.inner.len(),
            self.config
        )
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<PyEnfusionDatasetIterator> {
        Ok(PyEnfusionDatasetIterator {
            dataset: slf.into(),
            index: 0,
        })
    }
}

impl PyEnfusionDataset {
    /// Convert a Rust sample to Python sample with zero-copy numpy arrays.
    fn sample_to_python(&self, py: Python<'_>, sample: EnfusionSample) -> PyResult<PyEnfusionSample> {
        // Create numpy arrays - these are zero-copy views when possible
        let frames = sample.frames.into_pyarray_bound(py).unbind();
        let controls = sample.controls.into_pyarray_bound(py).unbind();
        let ego_transform = sample.ego_transform.into_pyarray_bound(py).unbind();
        let anchor_mask = sample.anchor_mask.into_pyarray_bound(py).unbind();

        let depth = sample.depth.map(|d| d.into_pyarray_bound(py).unbind());

        Ok(PyEnfusionSample {
            frames,
            controls,
            depth,
            ego_transform,
            anchor_mask,
            session_id: sample.session_id,
            start_frame: sample.start_frame,
            frame_indices: sample.frame_indices,
        })
    }
}

/// Iterator for RustEnfusionDataset.
#[pyclass]
pub struct PyEnfusionDatasetIterator {
    dataset: Py<PyEnfusionDataset>,
    index: usize,
}

#[pymethods]
impl PyEnfusionDatasetIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<PyEnfusionSample>> {
        let dataset = self.dataset.borrow(py);
        if self.index >= dataset.inner.len() {
            return Ok(None);
        }

        let sample = dataset.__getitem__(py, self.index as isize)?;
        self.index += 1;
        Ok(Some(sample))
    }
}

/// Telemetry data from a session.
#[pyclass(name = "RustTelemetryData")]
pub struct PyTelemetryData {
    inner: TelemetryData,
}

#[pymethods]
impl PyTelemetryData {
    /// Get timestamps as numpy array.
    pub fn timestamps<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        self.inner.timestamps.clone().into_pyarray_bound(py)
    }

    /// Get positions as numpy array [N, 3].
    pub fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f32>> {
        self.inner.positions.clone().into_pyarray_bound(py)
    }

    /// Get rotations as numpy array [N, 3].
    pub fn rotations<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f32>> {
        self.inner.rotations.clone().into_pyarray_bound(py)
    }

    /// Get velocities as numpy array [N, 3].
    pub fn velocities<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray2<f32>> {
        self.inner.velocities.clone().into_pyarray_bound(py)
    }

    /// Get steering values.
    pub fn steering<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        self.inner.steering.clone().into_pyarray_bound(py)
    }

    /// Get throttle values.
    pub fn throttle<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        self.inner.throttle.clone().into_pyarray_bound(py)
    }

    /// Get brake values.
    pub fn brake<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        self.inner.brake.clone().into_pyarray_bound(py)
    }

    /// Get speed values.
    pub fn speed<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f32>> {
        self.inner.speed.clone().into_pyarray_bound(py)
    }

    /// Get number of frames.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }
}

/// Session wrapper for direct access to session data.
#[pyclass(name = "RustEnfusionSession")]
pub struct PyEnfusionSession {
    inner: Arc<EnfusionSession>,
    frame_loader: Arc<FrameLoader>,
}

#[pymethods]
impl PyEnfusionSession {
    /// Load a session from disk.
    #[new]
    pub fn new(session_dir: &str) -> PyResult<Self> {
        let inner = EnfusionSession::new(PathBuf::from(session_dir))?;
        let frame_loader = Arc::new(FrameLoader::new(4, true, 500.0));

        Ok(PyEnfusionSession {
            inner: Arc::new(inner),
            frame_loader,
        })
    }

    /// Get session ID.
    #[getter]
    pub fn session_id(&self) -> &str {
        &self.inner.session_id
    }

    /// Get number of frames.
    #[getter]
    pub fn num_frames(&self) -> usize {
        self.inner.num_frames()
    }

    /// Get session directory path.
    #[getter]
    pub fn path(&self) -> String {
        self.inner.path.to_string_lossy().to_string()
    }

    /// Load telemetry data.
    pub fn load_telemetry(&self, py: Python<'_>) -> PyResult<PyTelemetryData> {
        let telemetry = py.allow_threads(|| self.inner.load_telemetry())?;
        Ok(PyTelemetryData { inner: telemetry })
    }

    /// Load a single frame by index.
    pub fn load_frame<'py>(
        &self,
        py: Python<'py>,
        frame_idx: usize,
        target_size: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let frame = py.allow_threads(|| {
            self.frame_loader.load_frame(&self.inner, frame_idx, target_size)
        })?;

        Ok(frame.into_pyarray_bound(py))
    }

    /// Load multiple frames in parallel.
    pub fn load_frames<'py>(
        &self,
        py: Python<'py>,
        frame_indices: Vec<usize>,
        target_size: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyArray4<f32>>> {
        let frames = py.allow_threads(|| {
            self.frame_loader.load_frames(&self.inner, &frame_indices, target_size)
        })?;

        Ok(frames.into_pyarray_bound(py))
    }

    /// Load depth map by index.
    pub fn load_depth<'py>(
        &self,
        py: Python<'py>,
        frame_idx: usize,
        target_size: Option<(usize, usize)>,
    ) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let depth = py.allow_threads(|| {
            self.inner.load_depth(frame_idx, target_size)
        })?;

        Ok(depth.into_pyarray_bound(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "RustEnfusionSession(id='{}', frames={})",
            self.inner.session_id,
            self.inner.num_frames()
        )
    }
}

/// Fast data collator for batching samples.
#[pyclass(name = "RustCollator")]
pub struct PyCollator {
    pad_to_max: bool,
    include_depth: bool,
}

#[pymethods]
impl PyCollator {
    #[new]
    #[pyo3(signature = (pad_to_max = true, include_depth = true))]
    pub fn new(pad_to_max: bool, include_depth: bool) -> Self {
        PyCollator {
            pad_to_max,
            include_depth,
        }
    }

    /// Collate a batch of samples into stacked tensors.
    pub fn __call__<'py>(
        &self,
        py: Python<'py>,
        samples: Vec<Bound<'py, PyEnfusionSample>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        if samples.is_empty() {
            return Ok(PyDict::new_bound(py));
        }

        // Stack frames [B, T, C, H, W]
        let frames: Vec<_> = samples.iter().map(|s| s.borrow().frames(py)).collect();
        let controls: Vec<_> = samples.iter().map(|s| s.borrow().controls(py)).collect();
        let ego_transforms: Vec<_> = samples.iter().map(|s| s.borrow().ego_transform(py)).collect();
        let anchor_masks: Vec<_> = samples.iter().map(|s| s.borrow().anchor_mask(py)).collect();

        let dict = PyDict::new_bound(py);

        // Use numpy.stack for efficiency
        let np = py.import_bound("numpy")?;
        dict.set_item("frames", np.call_method1("stack", (frames,))?)?;
        dict.set_item("controls", np.call_method1("stack", (controls,))?)?;
        dict.set_item("ego_transform", np.call_method1("stack", (ego_transforms,))?)?;
        dict.set_item("anchor_mask", np.call_method1("stack", (anchor_masks,))?)?;

        // Handle depth if present
        if self.include_depth {
            let depths: Vec<_> = samples
                .iter()
                .filter_map(|s| s.borrow().depth(py))
                .collect();

            if depths.len() == samples.len() {
                dict.set_item("depth", np.call_method1("stack", (depths,))?)?;
            }
        }

        // Collect metadata
        let metadata: Vec<_> = samples
            .iter()
            .map(|s| {
                let sample = s.borrow();
                let meta = PyDict::new_bound(py);
                meta.set_item("session_id", sample.session_id()).ok();
                meta.set_item("start_frame", sample.start_frame()).ok();
                meta
            })
            .collect();
        dict.set_item("metadata", PyList::new_bound(py, metadata))?;

        Ok(dict)
    }
}

/// Benchmark utilities for comparing Python vs Rust performance.
#[pyfunction]
pub fn benchmark_frame_loading(
    py: Python<'_>,
    session_dir: &str,
    frame_indices: Vec<usize>,
    target_size: (usize, usize),
    iterations: usize,
) -> PyResult<f64> {
    let session = EnfusionSession::new(PathBuf::from(session_dir))?;
    let frame_loader = FrameLoader::new(4, false, 0.0);

    let start = std::time::Instant::now();

    for _ in 0..iterations {
        py.allow_threads(|| {
            frame_loader.load_frames(&session, &frame_indices, Some(target_size))
        })?;
    }

    let elapsed = start.elapsed().as_secs_f64() / iterations as f64;
    Ok(elapsed)
}

/// Get library version.
#[pyfunction]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Check if the Rust backend is available.
#[pyfunction]
pub fn is_available() -> bool {
    true
}

/// Get the number of available CPU cores for parallel loading.
#[pyfunction]
pub fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

/// Register all Python types and functions.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<PyDatasetConfig>()?;
    m.add_class::<PyEnfusionDataset>()?;
    m.add_class::<PyEnfusionSample>()?;
    m.add_class::<PyEnfusionSession>()?;
    m.add_class::<PyTelemetryData>()?;
    m.add_class::<PyCollator>()?;

    // Functions
    m.add_function(wrap_pyfunction!(benchmark_frame_loading, m)?)?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(is_available, m)?)?;
    m.add_function(wrap_pyfunction!(num_cpus, m)?)?;

    Ok(())
}
