//! High-performance dataset for DriveDiT training.
//!
//! Provides a PyTorch-compatible dataset interface with parallel data loading,
//! caching, and efficient memory management.

use crate::error::{BatchError, Result};
use crate::session::{find_sessions, EnfusionSession};
use drivedit_types::BatchConfig;
use ndarray::{Array2, Array4};
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Sample index mapping: (session_idx, sample_idx_in_session).
#[derive(Debug, Clone, Copy)]
struct SampleIndex {
    session_idx: usize,
    sample_idx: usize,
}

/// High-performance dataset for DriveDiT training.
pub struct EnfusionDataset {
    /// Dataset configuration.
    config: BatchConfig,
    /// Session paths.
    session_paths: Vec<PathBuf>,
    /// Loaded sessions (lazy loaded).
    sessions: Arc<RwLock<HashMap<usize, EnfusionSession>>>,
    /// Sample index mapping.
    sample_indices: Vec<SampleIndex>,
    /// Total number of samples.
    total_samples: usize,
    /// Shuffled indices (for shuffling).
    shuffled_indices: Arc<RwLock<Vec<usize>>>,
    /// Random seed.
    seed: u64,
    /// Current epoch (for reshuffling).
    epoch: usize,
}

impl EnfusionDataset {
    /// Create a new EnfusionDataset from a root directory.
    pub fn from_path(root_path: &Path, config: BatchConfig) -> Result<Self> {
        let session_paths = find_sessions(root_path)?;

        if session_paths.is_empty() {
            return Err(BatchError::Session(format!(
                "No sessions found in {:?}",
                root_path
            )));
        }

        Self::from_session_paths(session_paths, config)
    }

    /// Create a dataset from a list of session paths.
    pub fn from_sessions(session_paths: Vec<String>, config: BatchConfig) -> Result<Self> {
        let paths: Vec<PathBuf> = session_paths.iter().map(PathBuf::from).collect();
        Self::from_session_paths(paths, config)
    }

    /// Create dataset from session paths.
    fn from_session_paths(session_paths: Vec<PathBuf>, config: BatchConfig) -> Result<Self> {
        // Probe sessions for sample counts (in parallel)
        let sample_counts: Vec<Result<usize>> = session_paths
            .par_iter()
            .map(|path| {
                let session = EnfusionSession::from_path(path, config.clone())?;
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
            session_paths,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            sample_indices,
            total_samples,
            shuffled_indices: Arc::new(RwLock::new(shuffled_indices)),
            seed,
            epoch: 0,
        };

        // Initial shuffle if enabled
        if dataset.config.shuffle {
            dataset.reshuffle();
        }

        Ok(dataset)
    }

    /// Get the total number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.total_samples
    }

    /// Check if dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.total_samples == 0
    }

    /// Get total number of samples.
    pub fn num_samples(&self) -> usize {
        self.total_samples
    }

    /// Get a sample by index.
    pub fn get(&self, idx: usize) -> Result<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)> {
        // Map index through shuffle if enabled
        let actual_idx = if self.config.shuffle {
            let indices = self.shuffled_indices.read();
            if idx >= indices.len() {
                return Err(BatchError::IndexOutOfBounds {
                    index: idx,
                    length: indices.len(),
                });
            }
            indices[idx]
        } else {
            idx
        };

        self.get_sample(actual_idx)
    }

    /// Get a batch of samples.
    pub fn get_batch(
        &self,
        indices: &[usize],
    ) -> Result<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)> {
        self.get_batch_inner(indices)
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
        if self.config.shuffle {
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
    pub fn preload_sessions(&self, session_indices: &[usize]) -> Result<usize> {
        let mut loaded = 0;
        for &idx in session_indices {
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

    /// Get or load a session.
    fn get_or_load_session(&self, session_idx: usize) -> Result<()> {
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
            .ok_or_else(|| BatchError::IndexOutOfBounds {
                index: session_idx,
                length: self.session_paths.len(),
            })?;

        let session = EnfusionSession::from_path(path, self.config.clone())?;

        // Store session
        {
            let mut sessions = self.sessions.write();
            sessions.insert(session_idx, session);
        }

        Ok(())
    }

    /// Get a single sample.
    fn get_sample(&self, idx: usize) -> Result<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)> {
        if idx >= self.total_samples {
            return Err(BatchError::IndexOutOfBounds {
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
            .ok_or_else(|| BatchError::Session("Session not loaded".to_string()))?;

        let frames = session.get_frames(sample_index.sample_idx)?;
        let controls = session.get_controls(sample_index.sample_idx)?;
        let depth = if self.config.load_depth {
            session.get_depth(sample_index.sample_idx)?
        } else {
            None
        };

        Ok((frames, controls, depth))
    }

    /// Get a batch of samples (parallel loading).
    fn get_batch_inner(
        &self,
        indices: &[usize],
    ) -> Result<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)> {
        if indices.is_empty() {
            return Err(BatchError::Config("Empty batch indices".to_string()));
        }

        // Load samples (sequential due to session locking)
        let mut samples = Vec::with_capacity(indices.len());
        for &idx in indices {
            samples.push(self.get_sample(idx)?);
        }

        // Combine into batched arrays
        let batch_size = indices.len();

        // Get shapes from first sample
        let (frame_shape, control_shape) = {
            let first = &samples[0];
            (first.0.shape().to_vec(), first.1.shape().to_vec())
        };

        // Create output arrays
        let t = frame_shape[0];
        let c = frame_shape[1];
        let h = frame_shape[2];
        let w = frame_shape[3];

        let mut frames = Array4::zeros((batch_size * t, c, h, w));
        let mut controls = Array2::zeros((batch_size * control_shape[0], control_shape[1]));
        let mut depth = if self.config.load_depth {
            Some(Array4::zeros((batch_size * t, 1, h, w)))
        } else {
            None
        };

        // Copy samples into batched arrays
        for (batch_idx, (f, ctrl, d)) in samples.into_iter().enumerate() {
            // Copy frames
            for ti in 0..t {
                for ci in 0..c {
                    for hi in 0..h {
                        for wi in 0..w {
                            frames[[batch_idx * t + ti, ci, hi, wi]] = f[[ti, ci, hi, wi]];
                        }
                    }
                }
            }

            // Copy controls
            for ti in 0..control_shape[0] {
                for di in 0..control_shape[1] {
                    controls[[batch_idx * control_shape[0] + ti, di]] = ctrl[[ti, di]];
                }
            }

            // Copy depth
            if let (Some(ref mut depth_out), Some(ref depth_in)) = (&mut depth, &d) {
                for ti in 0..t {
                    for hi in 0..h {
                        for wi in 0..w {
                            depth_out[[batch_idx * t + ti, 0, hi, wi]] = depth_in[[ti, 0, hi, wi]];
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

impl std::fmt::Display for EnfusionDataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "EnfusionDataset(samples={}, sessions={}, seq_len={})",
            self.total_samples,
            self.session_paths.len(),
            self.config.sequence_length
        )
    }
}

/// Iterator over dataset samples.
pub struct DatasetIterator<'a> {
    dataset: &'a EnfusionDataset,
    current_idx: usize,
}

impl<'a> DatasetIterator<'a> {
    /// Create a new iterator.
    pub fn new(dataset: &'a EnfusionDataset) -> Self {
        Self {
            dataset,
            current_idx: 0,
        }
    }
}

impl<'a> Iterator for DatasetIterator<'a> {
    type Item = Result<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.len() {
            return None;
        }

        let result = self.dataset.get(self.current_idx);
        self.current_idx += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.dataset.len() - self.current_idx;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for DatasetIterator<'a> {}

/// Batch iterator over dataset samples.
pub struct BatchIterator<'a> {
    dataset: &'a EnfusionDataset,
    batch_size: usize,
    drop_last: bool,
    current_batch: usize,
    num_batches: usize,
}

impl<'a> BatchIterator<'a> {
    /// Create a new batch iterator.
    pub fn new(dataset: &'a EnfusionDataset, batch_size: usize, drop_last: bool) -> Self {
        let total = dataset.len();
        let num_batches = if drop_last {
            total / batch_size
        } else {
            (total + batch_size - 1) / batch_size
        };

        Self {
            dataset,
            batch_size,
            drop_last,
            current_batch: 0,
            num_batches,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = Result<(Array4<f32>, Array2<f32>, Option<Array4<f32>>)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_batch >= self.num_batches {
            return None;
        }

        let start = self.current_batch * self.batch_size;
        let end = (start + self.batch_size).min(self.dataset.len());

        if self.drop_last && end - start < self.batch_size {
            return None;
        }

        let indices: Vec<usize> = (start..end).collect();
        self.current_batch += 1;

        Some(self.dataset.get_batch(&indices))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.num_batches - self.current_batch;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for BatchIterator<'a> {}

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
        let mut dataset1_indices: Vec<usize> = (0..100).collect();
        let mut dataset2_indices: Vec<usize> = (0..100).collect();

        // Fisher-Yates with same seed should produce same result
        let seed: u64 = 42;
        let epoch: usize = 0;

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        epoch.hash(&mut hasher);
        let combined_seed = hasher.finish();

        // Shuffle first
        let mut rng_state = combined_seed;
        for i in (1..dataset1_indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            dataset1_indices.swap(i, j);
        }

        // Shuffle second with same seed
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        epoch.hash(&mut hasher);
        let combined_seed = hasher.finish();

        let mut rng_state = combined_seed;
        for i in (1..dataset2_indices.len()).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            dataset2_indices.swap(i, j);
        }

        assert_eq!(dataset1_indices, dataset2_indices);
    }
}
