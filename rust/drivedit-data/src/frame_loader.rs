//! Frame loader service with caching and parallel loading.
//!
//! Provides high-performance frame loading with:
//! - LRU cache for frequently accessed frames
//! - Parallel batch loading with rayon
//! - Memory-mapped I/O for large datasets
//! - Zero-copy output for numpy integration

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ndarray::{Array3, Array4};
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::dataset::EnfusionSession;
use crate::error::{DataError, DataResult};
use crate::frame::{FrameLoader, FrameLoaderConfig};

/// LRU cache entry for decoded frames.
struct CacheEntry {
    data: Array3<f32>,
    access_count: u64,
}

/// Frame loader service with caching and statistics.
pub struct FrameLoaderService {
    loader_config: FrameLoaderConfig,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_cache_mb: f64,
    current_cache_mb: Arc<RwLock<f64>>,
    access_counter: Arc<RwLock<u64>>,
    cache_enabled: bool,

    // Statistics
    cache_hits: Arc<RwLock<u64>>,
    cache_misses: Arc<RwLock<u64>>,
    total_loads: Arc<RwLock<u64>>,
}

impl FrameLoaderService {
    /// Create a new frame loader service.
    pub fn new(num_workers: usize, cache_enabled: bool, max_cache_mb: f64) -> Self {
        let loader_config = FrameLoaderConfig {
            target_width: 256,
            target_height: 256,
            num_workers,
            ..Default::default()
        };

        FrameLoaderService {
            loader_config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            max_cache_mb,
            current_cache_mb: Arc::new(RwLock::new(0.0)),
            access_counter: Arc::new(RwLock::new(0)),
            cache_enabled,
            cache_hits: Arc::new(RwLock::new(0)),
            cache_misses: Arc::new(RwLock::new(0)),
            total_loads: Arc::new(RwLock::new(0)),
        }
    }

    /// Load a single frame from a session.
    pub fn load_frame(
        &self,
        session: &EnfusionSession,
        frame_idx: usize,
        target_size: Option<(usize, usize)>,
    ) -> DataResult<Array3<f32>> {
        let frame_path = session.frame_path(frame_idx).ok_or_else(|| {
            DataError::IndexOutOfBounds {
                index: frame_idx,
                length: session.num_frames(),
            }
        })?;

        let cache_key = self.make_cache_key(frame_path, target_size);

        // Check cache first
        if self.cache_enabled {
            if let Some(frame) = self.get_cached(&cache_key) {
                *self.cache_hits.write() += 1;
                return Ok(frame);
            }
            *self.cache_misses.write() += 1;
        }

        *self.total_loads.write() += 1;

        // Load frame from disk
        let frame = self.load_frame_from_disk(frame_path, target_size)?;

        // Cache the result
        if self.cache_enabled {
            self.cache_frame(&cache_key, frame.clone());
        }

        Ok(frame)
    }

    /// Load multiple frames from a session in parallel.
    pub fn load_frames(
        &self,
        session: &EnfusionSession,
        frame_indices: &[usize],
        target_size: Option<(usize, usize)>,
    ) -> DataResult<Array4<f32>> {
        let (h, w) = target_size.unwrap_or((256, 256));
        let seq_len = frame_indices.len();

        // Load frames in parallel
        let frames: Vec<DataResult<Array3<f32>>> = frame_indices
            .par_iter()
            .map(|&idx| self.load_frame(session, idx, target_size))
            .collect();

        // Stack into 4D array
        let mut result = Array4::zeros((seq_len, 3, h, w));

        for (i, frame_result) in frames.into_iter().enumerate() {
            let frame = frame_result?;
            for c in 0..3 {
                for y in 0..h {
                    for x in 0..w {
                        result[[i, c, y, x]] = frame[[c, y, x]];
                    }
                }
            }
        }

        Ok(result)
    }

    /// Prefetch frames into cache.
    pub fn prefetch_frames(
        &self,
        session: &EnfusionSession,
        frame_indices: &[usize],
        target_size: Option<(usize, usize)>,
    ) -> DataResult<()> {
        if !self.cache_enabled {
            return Ok(());
        }

        // Load frames in parallel (results will be cached)
        frame_indices.par_iter().for_each(|&idx| {
            let _ = self.load_frame(session, idx, target_size);
        });

        Ok(())
    }

    /// Get cache statistics.
    pub fn cache_stats(&self) -> HashMap<String, f64> {
        let hits = *self.cache_hits.read() as f64;
        let misses = *self.cache_misses.read() as f64;
        let total = hits + misses;
        let hit_rate = if total > 0.0 { hits / total } else { 0.0 };

        let mut stats = HashMap::new();
        stats.insert("cache_hits".to_string(), hits);
        stats.insert("cache_misses".to_string(), misses);
        stats.insert("cache_hit_rate".to_string(), hit_rate);
        stats.insert("cache_size_mb".to_string(), *self.current_cache_mb.read());
        stats.insert("cache_max_mb".to_string(), self.max_cache_mb);
        stats.insert("total_loads".to_string(), *self.total_loads.read() as f64);
        stats
    }

    /// Clear the frame cache.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        cache.clear();
        *self.current_cache_mb.write() = 0.0;
        *self.cache_hits.write() = 0;
        *self.cache_misses.write() = 0;
    }

    /// Make a cache key from path and size.
    fn make_cache_key(&self, path: &Path, target_size: Option<(usize, usize)>) -> String {
        let (h, w) = target_size.unwrap_or((0, 0));
        format!("{}:{}x{}", path.display(), h, w)
    }

    /// Get a frame from cache.
    fn get_cached(&self, key: &str) -> Option<Array3<f32>> {
        let mut cache = self.cache.write();
        if let Some(entry) = cache.get_mut(key) {
            let mut counter = self.access_counter.write();
            *counter += 1;
            entry.access_count = *counter;
            return Some(entry.data.clone());
        }
        None
    }

    /// Cache a frame.
    fn cache_frame(&self, key: &str, frame: Array3<f32>) {
        let frame_size_mb = (frame.len() * 4) as f64 / (1024.0 * 1024.0);

        // Evict if necessary
        while *self.current_cache_mb.read() + frame_size_mb > self.max_cache_mb {
            if !self.evict_lru() {
                break;
            }
        }

        // Add to cache
        let mut cache = self.cache.write();
        let mut counter = self.access_counter.write();
        *counter += 1;

        cache.insert(
            key.to_string(),
            CacheEntry {
                data: frame,
                access_count: *counter,
            },
        );

        *self.current_cache_mb.write() += frame_size_mb;
    }

    /// Evict least recently used entry.
    fn evict_lru(&self) -> bool {
        let mut cache = self.cache.write();

        if cache.is_empty() {
            return false;
        }

        // Find LRU entry
        let lru_key = cache
            .iter()
            .min_by_key(|(_, entry)| entry.access_count)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            if let Some(entry) = cache.remove(&key) {
                let size_mb = (entry.data.len() * 4) as f64 / (1024.0 * 1024.0);
                *self.current_cache_mb.write() -= size_mb;
                return true;
            }
        }

        false
    }

    /// Load a frame from disk.
    fn load_frame_from_disk(
        &self,
        path: &Path,
        target_size: Option<(usize, usize)>,
    ) -> DataResult<Array3<f32>> {
        // Create frame loader with target size
        let (h, w) = target_size.unwrap_or((256, 256));
        let config = FrameLoaderConfig {
            target_width: w as u32,
            target_height: h as u32,
            ..self.loader_config.clone()
        };

        let mut loader = FrameLoader::new(config);
        let frame = loader
            .load_frame(path)
            .map_err(|e| DataError::Frame(e.to_string()))?;

        // Convert to ndarray
        let (width, height, channels) = frame.shape();
        let data = frame.as_slice();

        let mut result = Array3::zeros((channels as usize, height as usize, width as usize));
        for c in 0..channels as usize {
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let idx = c * (height as usize * width as usize) + y * width as usize + x;
                    result[[c, y, x]] = data[idx];
                }
            }
        }

        Ok(result)
    }
}

// Alias for backward compatibility with python.rs
pub type FrameLoader = FrameLoaderService;

impl FrameLoader {
    /// Compatibility wrapper for python.rs
    pub fn load_frame_compat(
        &self,
        session: &EnfusionSession,
        frame_idx: usize,
        target_size: Option<(usize, usize)>,
    ) -> DataResult<Array3<f32>> {
        self.load_frame(session, frame_idx, target_size)
    }

    /// Compatibility wrapper for python.rs
    pub fn load_frames_compat(
        &self,
        session: &EnfusionSession,
        frame_indices: &[usize],
        target_size: Option<(usize, usize)>,
    ) -> DataResult<Array4<f32>> {
        self.load_frames(session, frame_indices, target_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_loader_creation() {
        let loader = FrameLoaderService::new(4, true, 1000.0);
        assert!(loader.cache_enabled);
        assert_eq!(loader.max_cache_mb, 1000.0);
    }

    #[test]
    fn test_cache_stats() {
        let loader = FrameLoaderService::new(4, true, 1000.0);
        let stats = loader.cache_stats();
        assert_eq!(stats.get("cache_hits"), Some(&0.0));
        assert_eq!(stats.get("cache_misses"), Some(&0.0));
    }
}
