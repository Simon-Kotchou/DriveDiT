//! High-performance dataset for DriveDiT training.
//!
//! Provides a PyTorch-compatible dataset interface with parallel data loading,
//! caching, and efficient memory management.

use crate::config::{DatasetConfig, LoaderConfig, TelemetryConfig};
use crate::error::{DataError, DataResult};
use crate::session::{find_sessions, EnfusionSession};
use ndarray::{Array2, Array4};
use numpy::{PyArray2, PyArray4, ToPyArray};
use parking_lot::RwLock;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Sample index mapping: (session_idx, sample_idx_in_session)
#[derive(Debug, Clone, Copy)]
struct SampleIndex {
    session_idx: usize,
    sample_idx: usize,
}

/// High-performance dataset for DriveDiT training.
#[pyclass]
pub struct EnfusionDataset {
    /// Dataset configuration
    config: DatasetConfig,
    /// Loader configuration
    loader_config: LoaderConfig,
    /// Telemetry configuration
    telemetry_config: TelemetryConfig,
    /// Session paths
    session_paths: Vec<PathBuf>,
    /// Loaded sessions (lazy loaded)
    sessions: Arc<RwLock<HashMap<usize, EnfusionSession>>>,
    /// Sample index mapping
    sample_indices: Vec<SampleIndex>,
    /// Total number of samples
    total_samples: usize,
    /// Shuffled indices (for shuffling)
    shuffled_indices: Arc<RwLock<Vec<usize>>>,
    /// Random seed
    seed: u64,
    /// Current epoch (for reshuffling)
    epoch: usize,
}

#[pymethods]
impl EnfusionDataset {
    /// Create a new EnfusionDataset from a root directory.
    #[new]
    #[pyo3(signature = (root_path, config = None, loader_config = None, telemetry_config = None))]
    pub fn new(
        root_path: &str,
        config: Option<DatasetConfig>,
        loader_config: Option<LoaderConfig>,
        telemetry_config: Option<TelemetryConfig>,
    ) -> PyResult<Self> {
        let config = config.unwrap_or_default();
        let loader_config = loader_config.unwrap_or_default();
        let telemetry_config = telemetry_config.unwrap_or_default();

        let dataset = Self::from_path(
            Path::new(root_path),
            config,
            loader_config,
            telemetry_config,
        )?;
        Ok(dataset)
    }

    /// Create a dataset from a list of session paths.
    #[staticmethod]
    #[pyo3(signature = (session_paths, config = None, loader_config = None, telemetry_config = None))]
    pub fn from_sessions(
        session_paths: Vec<String>,
        config: Option<DatasetConfig>,
        loader_config: Option<LoaderConfig>,
        telemetry_config: Option<TelemetryConfig>,
    ) -> PyResult<Self> {
        let config = config.unwrap_or_default();
        let loader_config = loader_config.unwrap_or_default();
        let telemetry_config = telemetry_config.unwrap_or_default();

        let paths: Vec<PathBuf> = session_paths.iter().map(PathBuf::from).collect();

        let dataset = Self::from_session_paths(paths, config, loader_config, telemetry_config)?;
        Ok(dataset)
    }

    /// Get the total number of samples in the dataset.
    pub fn __len__(&self) -> usize {
        self.total_samples
    }

    /// Get a sample by index.
    pub fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        idx: usize,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )> {
        // Map index through shuffle if enabled
        let actual_idx = if self.loader_config.shuffle {
            let indices = self.shuffled_indices.read();
            if idx >= indices.len() {
                return Err(DataError::IndexOutOfBounds {
                    index: idx,
                    length: indices.len(),
                }
                .into());
            }
            indices[idx]
        } else {
            idx
        };

        let sample = self.get_sample(actual_idx)?;

        Ok((
            sample.0.to_pyarray(py),
            sample.1.to_pyarray(py),
            sample.2.map(|d| d.to_pyarray(py)),
        ))
    }

    /// Get a batch of samples.
    pub fn get_batch<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )> {
        let samples = self.get_batch_inner(&indices)?;
        Ok((
            samples.0.to_pyarray(py),
            samples.1.to_pyarray(py),
            samples.2.map(|d| d.to_pyarray(py)),
        ))
    }

    /// Get the number of sessions.
    pub fn num_sessions(&self) -> usize {
        self.session_paths.len()
    }

    /// Get session paths.
    pub fn session_paths(&self) -> Vec<String> {
        self.session_paths
            .iter()
            .map(|p| p.to_string_lossy().to_string())
            .collect()
    }

    /// Set the current epoch (triggers reshuffle).
    pub fn set_epoch(&mut self, epoch: usize) {
        self.epoch = epoch;
        if self.loader_config.shuffle {
            self.reshuffle();
        }
    }

    /// Get the current epoch.
    pub fn get_epoch(&self) -> usize {
        self.epoch
    }

    /// Shuffle the dataset indices.
    pub fn shuffle(&mut self) {
        self.reshuffle();
    }

    /// Preload sessions into memory.
    pub fn preload_sessions(&self, session_indices: Vec<usize>) -> PyResult<usize> {
        let mut loaded = 0;
        for idx in session_indices {
            if idx < self.session_paths.len() {
                self.get_or_load_session(idx)?;
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    /// Clear all session caches.
    pub fn clear_cache(&self) {
        let mut sessions = self.sessions.write();
        for session in sessions.values_mut() {
            session.clear_cache();
        }
    }

    /// Unload a specific session from memory.
    pub fn unload_session(&self, session_idx: usize) {
        let mut sessions = self.sessions.write();
        sessions.remove(&session_idx);
    }

    /// Get dataset statistics.
    pub fn stats(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();
        stats.insert("total_samples".to_string(), self.total_samples);
        stats.insert("num_sessions".to_string(), self.session_paths.len());
        stats.insert("loaded_sessions".to_string(), self.sessions.read().len());
        stats.insert("sequence_length".to_string(), self.config.sequence_length);
        stats.insert("context_frames".to_string(), self.config.context_frames);
        stats
    }

    fn __repr__(&self) -> String {
        format!(
            "EnfusionDataset(samples={}, sessions={}, seq_len={})",
            self.total_samples,
            self.session_paths.len(),
            self.config.sequence_length
        )
    }
}

impl EnfusionDataset {
    /// Create dataset from a root path.
    fn from_path(
        root: &Path,
        config: DatasetConfig,
        loader_config: LoaderConfig,
        telemetry_config: TelemetryConfig,
    ) -> DataResult<Self> {
        let session_paths = find_sessions(root)?;

        if session_paths.is_empty() {
            return Err(DataError::Session(format!(
                "No sessions found in {:?}",
                root
            )));
        }

        Self::from_session_paths(session_paths, config, loader_config, telemetry_config)
    }

    /// Create dataset from session paths.
    fn from_session_paths(
        session_paths: Vec<PathBuf>,
        config: DatasetConfig,
        loader_config: LoaderConfig,
        telemetry_config: TelemetryConfig,
    ) -> DataResult<Self> {
        // Probe sessions for sample counts (in parallel)
        let sample_counts: Vec<DataResult<usize>> = session_paths
            .par_iter()
            .map(|path| {
                let session = EnfusionSession::from_path(
                    path,
                    config.clone(),
                    telemetry_config.clone(),
                )?;
                Ok(session.num_samples())
            })
            .collect();

        // Build sample index mapping
        let mut sample_indices = Vec::new();
        let mut total_samples = 0;

        for (session_idx, result) in sample_counts.into_iter().enumerate() {
            let count = result?;
            for sample_idx in 0..count {
                sample_indices.push(SampleIndex {
                    session_idx,
                    sample_idx,
                });
            }
            total_samples += count;
        }

        // Initialize shuffled indices
        let shuffled_indices: Vec<usize> = (0..total_samples).collect();
        let seed = config.seed;

        let mut dataset = Self {
            config,
            loader_config,
            telemetry_config,
            session_paths,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            sample_indices,
            total_samples,
            shuffled_indices: Arc::new(RwLock::new(shuffled_indices)),
            seed,
            epoch: 0,
        };

        // Initial shuffle if enabled
        if dataset.loader_config.shuffle {
            dataset.reshuffle();
        }

        Ok(dataset)
    }

    /// Get or load a session.
    fn get_or_load_session(&self, session_idx: usize) -> DataResult<()> {
        // Check if already loaded
        {
            let sessions = self.sessions.read();
            if sessions.contains_key(&session_idx) {
                return Ok(());
            }
        }

        // Load session
        let path = self
            .session_paths
            .get(session_idx)
            .ok_or_else(|| DataError::IndexOutOfBounds {
                index: session_idx,
                length: self.session_paths.len(),
            })?;

        let session = EnfusionSession::from_path(
            path,
            self.config.clone(),
            self.telemetry_config.clone(),
        )?;

        // Store session
        {
            let mut sessions = self.sessions.write();
            sessions.insert(session_idx, session);
        }

        Ok(())
    }

    /// Get a single sample.
    fn get_sample(&self, idx: usize) -> DataResult<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)> {
        if idx >= self.total_samples {
            return Err(DataError::IndexOutOfBounds {
                index: idx,
                length: self.total_samples,
            });
        }

        let sample_index = &self.sample_indices[idx];

        // Ensure session is loaded
        self.get_or_load_session(sample_index.session_idx)?;

        // Get sample from session
        let mut sessions = self.sessions.write();
        let session = sessions
            .get_mut(&sample_index.session_idx)
            .ok_or_else(|| DataError::Session("Session not loaded".to_string()))?;

        let frames = session.load_sample_frames(sample_index.sample_idx)?;
        let controls = session.load_sample_controls(sample_index.sample_idx)?;
        let depth = if self.config.load_depth {
            Some(session.load_sample_depth(sample_index.sample_idx)?)
        } else {
            None
        };

        Ok((frames, controls, depth))
    }

    /// Get a batch of samples (parallel loading).
    fn get_batch_inner(
        &self,
        indices: &[usize],
    ) -> DataResult<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)> {
        if indices.is_empty() {
            return Err(DataError::Config("Empty batch indices".to_string()));
        }

        // Load samples in parallel
        let samples: Vec<DataResult<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)>> = indices
            .par_iter()
            .map(|&idx| self.get_sample(idx))
            .collect();

        // Combine into batched arrays
        let batch_size = indices.len();
        let first_sample = samples.first().ok_or_else(|| {
            DataError::Config("No samples loaded".to_string())
        })?.as_ref().map_err(|e| DataError::Frame(e.to_string()))?;

        let frame_shape = first_sample.0.shape();
        let control_shape = first_sample.1.shape();

        // Create output arrays
        let mut frames = Array4::zeros((
            batch_size * frame_shape[0],
            frame_shape[1],
            frame_shape[2],
            frame_shape[3],
        ));
        let mut controls = Array2::zeros((batch_size * control_shape[0], control_shape[1]));
        let mut depth = if self.config.load_depth {
            Some(Array4::zeros((
                batch_size * frame_shape[0],
                1,
                frame_shape[2],
                frame_shape[3],
            )))
        } else {
            None
        };

        // Copy samples into batched arrays
        for (batch_idx, result) in samples.into_iter().enumerate() {
            let (f, c, d) = result?;
            let t = f.shape()[0];

            // Copy frames
            for t_idx in 0..t {
                for ch in 0..frame_shape[1] {
                    for h in 0..frame_shape[2] {
                        for w in 0..frame_shape[3] {
                            frames[[batch_idx * t + t_idx, ch, h, w]] = f[[t_idx, ch, h, w]];
                        }
                    }
                }
            }

            // Copy controls
            for t_idx in 0..control_shape[0] {
                for dim in 0..control_shape[1] {
                    controls[[batch_idx * control_shape[0] + t_idx, dim]] = c[[t_idx, dim]];
                }
            }

            // Copy depth
            if let (Some(ref mut depth_out), Some(ref depth_in)) = (&mut depth, &d) {
                for t_idx in 0..t {
                    for h in 0..frame_shape[2] {
                        for w in 0..frame_shape[3] {
                            depth_out[[batch_idx * t + t_idx, 0, h, w]] = depth_in[[t_idx, 0, h, w]];
                        }
                    }
                }
            }
        }

        Ok((frames, controls, depth))
    }

    /// Reshuffle indices with current seed and epoch.
    fn reshuffle(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.seed.hash(&mut hasher);
        self.epoch.hash(&mut hasher);
        let combined_seed = hasher.finish();

        let mut indices = self.shuffled_indices.write();

        // Fisher-Yates shuffle with seeded PRNG
        let mut rng_state = combined_seed;
        for i in (1..indices.len()).rev() {
            // Simple LCG for deterministic shuffling
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            indices.swap(i, j);
        }
    }
}

/// Create a data loader iterator.
#[pyclass]
pub struct DataLoaderIterator {
    dataset: Py<EnfusionDataset>,
    batch_size: usize,
    drop_last: bool,
    current_idx: usize,
    num_batches: usize,
}

#[pymethods]
impl DataLoaderIterator {
    #[new]
    pub fn new(dataset: Py<EnfusionDataset>, batch_size: usize, drop_last: bool) -> Self {
        Python::with_gil(|py| {
            let ds = dataset.borrow(py);
            let total = ds.total_samples;
            let num_batches = if drop_last {
                total / batch_size
            } else {
                (total + batch_size - 1) / batch_size
            };

            Self {
                dataset,
                batch_size,
                drop_last,
                current_idx: 0,
                num_batches,
            }
        })
    }

    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__<'py>(
        mut slf: PyRefMut<'py, Self>,
        py: Python<'py>,
    ) -> PyResult<Option<(
        Bound<'py, PyArray4<f32>>,
        Bound<'py, PyArray2<f32>>,
        Option<Bound<'py, PyArray4<f32>>>,
    )>> {
        if slf.current_idx >= slf.num_batches {
            return Ok(None);
        }

        let start = slf.current_idx * slf.batch_size;
        let ds = slf.dataset.borrow(py);
        let end = (start + slf.batch_size).min(ds.total_samples);

        if slf.drop_last && end - start < slf.batch_size {
            return Ok(None);
        }

        let indices: Vec<usize> = (start..end).collect();
        slf.current_idx += 1;

        let batch = ds.get_batch(py, indices)?;
        Ok(Some(batch))
    }

    fn __len__(&self) -> usize {
        self.num_batches
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_index() {
        let idx = SampleIndex {
            session_idx: 5,
            sample_idx: 10,
        };
        assert_eq!(idx.session_idx, 5);
        assert_eq!(idx.sample_idx, 10);
    }

    #[test]
    fn test_shuffle_determinism() {
        // Test that shuffling with same seed produces same order
        let indices1: Vec<usize> = (0..100).collect();
        let indices2: Vec<usize> = (0..100).collect();

        // With same seed and epoch, shuffle should be identical
        // (Would need actual implementation to test)
    }
}
