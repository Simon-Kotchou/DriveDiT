//! Parallel frame loading with caching and resizing.
//!
//! This module provides high-performance frame loading for DriveDiT training,
//! with support for parallel loading, LRU caching, and on-the-fly resizing.

use crate::error::{BatchError, Result};
use crate::simd::normalize_frame;
use drivedit_io::{decode_auto, decode_bmp};
use drivedit_types::{BatchConfig, NormalizationParams, ProcessedFrame, RawFrame};
use fast_image_resize::{
    FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer, images::Image,
};
use ndarray::Array4;
use parking_lot::RwLock;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// High-performance frame loader with caching and parallel loading.
pub struct FrameLoader {
    /// Configuration.
    config: BatchConfig,
    /// Normalization parameters.
    norm_params: NormalizationParams,
    /// LRU cache for loaded frames.
    cache: Arc<RwLock<HashMap<String, ProcessedFrame>>>,
    /// Maximum cache size.
    cache_size: usize,
    /// Cache access order for LRU eviction.
    cache_order: Arc<RwLock<Vec<String>>>,
    /// Resizer instance (thread-local).
    resizer: Resizer,
}

impl FrameLoader {
    /// Create a new frame loader.
    pub fn new(config: BatchConfig) -> Self {
        Self::with_params(config, NormalizationParams::default())
    }

    /// Create a frame loader with custom normalization.
    pub fn with_params(config: BatchConfig, norm_params: NormalizationParams) -> Self {
        let cache_size = config.cache_size_mb * 1024 * 1024
            / (config.target_width as usize * config.target_height as usize * 3 * 4);

        Self {
            config,
            norm_params,
            cache: Arc::new(RwLock::new(HashMap::new())),
            cache_size: cache_size.max(16),
            cache_order: Arc::new(RwLock::new(Vec::new())),
            resizer: Resizer::new(),
        }
    }

    /// Load a single frame from a file path.
    pub fn load_frame(&mut self, path: &str) -> Result<ProcessedFrame> {
        // Check cache
        {
            let cache = self.cache.read();
            if let Some(frame) = cache.get(path) {
                return Ok(frame.clone());
            }
        }

        // Load and process frame
        let frame = self.load_and_process(path)?;

        // Update cache
        self.cache_frame(path.to_string(), frame.clone());

        Ok(frame)
    }

    /// Load multiple frames in parallel.
    pub fn load_frames(&self, paths: &[String]) -> Result<Vec<ProcessedFrame>> {
        paths
            .par_iter()
            .map(|path| {
                // Check cache first
                {
                    let cache = self.cache.read();
                    if let Some(frame) = cache.get(path) {
                        return Ok(frame.clone());
                    }
                }

                // Load and process
                self.load_and_process_parallel(path)
            })
            .collect()
    }

    /// Load frames and return as ndarray [T, C, H, W].
    pub fn load_frames_array(&self, paths: &[String]) -> Result<Array4<f32>> {
        let frames = self.load_frames(paths)?;

        if frames.is_empty() {
            return Err(BatchError::Frame("No frames to load".to_string()));
        }

        let t = frames.len();
        let (c, h, w) = frames[0].shape();

        let mut array = Array4::zeros((t, c as usize, h as usize, w as usize));

        for (i, frame) in frames.iter().enumerate() {
            let pixels = frame.as_slice();
            for ch in 0..c as usize {
                for y in 0..h as usize {
                    for x in 0..w as usize {
                        let idx = ch * (h as usize * w as usize) + y * w as usize + x;
                        array[[i, ch, y, x]] = pixels[idx];
                    }
                }
            }
        }

        Ok(array)
    }

    /// Preload frames into cache.
    pub fn preload(&self, paths: Vec<String>) -> Result<usize> {
        let loaded: Vec<Result<ProcessedFrame>> = paths
            .par_iter()
            .map(|path| self.load_and_process_parallel(path))
            .collect();

        let mut count = 0;
        let mut cache = self.cache.write();
        let mut order = self.cache_order.write();

        for (path, result) in paths.into_iter().zip(loaded.into_iter()) {
            if let Ok(frame) = result {
                if cache.len() >= self.cache_size {
                    // Evict oldest
                    if let Some(old) = order.first().cloned() {
                        cache.remove(&old);
                        order.remove(0);
                    }
                }
                cache.insert(path.clone(), frame);
                order.push(path);
                count += 1;
            }
        }

        Ok(count)
    }

    /// Clear the frame cache.
    pub fn clear_cache(&self) {
        let mut cache = self.cache.write();
        let mut order = self.cache_order.write();
        cache.clear();
        order.clear();
    }

    /// Get current cache size.
    pub fn cache_len(&self) -> usize {
        self.cache.read().len()
    }

    /// Load and process a frame (single-threaded version).
    fn load_and_process(&mut self, path: &str) -> Result<ProcessedFrame> {
        let path_ref = Path::new(path);

        // Read file
        let data = fs::read(path_ref)?;

        // Decode based on extension
        let raw_frame = if path.ends_with(".bmp") {
            decode_bmp(&data)?
        } else {
            decode_auto(&data)?
        };

        // Resize if needed
        let resized = self.resize_frame(&raw_frame)?;

        // Normalize to CHW format
        let processed = normalize_frame(&resized, &self.norm_params);

        Ok(processed)
    }

    /// Load and process a frame (thread-safe version for parallel loading).
    fn load_and_process_parallel(&self, path: &str) -> Result<ProcessedFrame> {
        let path_ref = Path::new(path);

        // Read file
        let data = fs::read(path_ref)?;

        // Decode based on extension
        let raw_frame = if path.ends_with(".bmp") {
            decode_bmp(&data)?
        } else {
            decode_auto(&data)?
        };

        // Resize if needed
        let resized = self.resize_frame_new_resizer(&raw_frame)?;

        // Normalize to CHW format
        let processed = normalize_frame(&resized, &self.norm_params);

        Ok(processed)
    }

    /// Resize a frame to target dimensions using the instance resizer.
    fn resize_frame(&mut self, frame: &RawFrame) -> Result<RawFrame> {
        let target_w = self.config.target_width;
        let target_h = self.config.target_height;

        // Skip if already at target size
        if frame.width == target_w && frame.height == target_h {
            return Ok(frame.clone());
        }

        // Create source image
        let src_image = Image::from_vec_u8(
            frame.width.try_into().unwrap(),
            frame.height.try_into().unwrap(),
            frame.pixels.clone(),
            PixelType::U8x3,
        ).map_err(|e| BatchError::Frame(format!("Failed to create source image: {}", e)))?;

        // Create destination image
        let mut dst_image = Image::new(
            target_w.try_into().unwrap(),
            target_h.try_into().unwrap(),
            PixelType::U8x3,
        );

        // Resize
        let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
        self.resizer.resize(&src_image, &mut dst_image, &options)
            .map_err(|e| BatchError::Frame(format!("Resize failed: {}", e)))?;

        Ok(RawFrame::new(
            frame.frame_id,
            frame.timestamp_ms,
            dst_image.into_vec(),
            target_w,
            target_h,
            frame.telemetry.clone(),
        ))
    }

    /// Resize a frame using a new resizer (thread-safe).
    fn resize_frame_new_resizer(&self, frame: &RawFrame) -> Result<RawFrame> {
        let target_w = self.config.target_width;
        let target_h = self.config.target_height;

        // Skip if already at target size
        if frame.width == target_w && frame.height == target_h {
            return Ok(frame.clone());
        }

        // Create source image
        let src_image = Image::from_vec_u8(
            frame.width.try_into().unwrap(),
            frame.height.try_into().unwrap(),
            frame.pixels.clone(),
            PixelType::U8x3,
        ).map_err(|e| BatchError::Frame(format!("Failed to create source image: {}", e)))?;

        // Create destination image
        let mut dst_image = Image::new(
            target_w.try_into().unwrap(),
            target_h.try_into().unwrap(),
            PixelType::U8x3,
        );

        // Create thread-local resizer
        let mut resizer = Resizer::new();
        let options = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FilterType::Bilinear));
        resizer.resize(&src_image, &mut dst_image, &options)
            .map_err(|e| BatchError::Frame(format!("Resize failed: {}", e)))?;

        Ok(RawFrame::new(
            frame.frame_id,
            frame.timestamp_ms,
            dst_image.into_vec(),
            target_w,
            target_h,
            frame.telemetry.clone(),
        ))
    }

    /// Add a frame to the cache.
    fn cache_frame(&self, path: String, frame: ProcessedFrame) {
        let mut cache = self.cache.write();
        let mut order = self.cache_order.write();

        // Evict if needed
        while cache.len() >= self.cache_size && !order.is_empty() {
            if let Some(old) = order.first().cloned() {
                cache.remove(&old);
                order.remove(0);
            }
        }

        cache.insert(path.clone(), frame);
        order.push(path);
    }
}

impl Clone for FrameLoader {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            norm_params: self.norm_params,
            cache: Arc::clone(&self.cache),
            cache_size: self.cache_size,
            cache_order: Arc::clone(&self.cache_order),
            resizer: Resizer::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_loader_creation() {
        let config = BatchConfig::minimal();
        let loader = FrameLoader::new(config);
        assert_eq!(loader.cache_len(), 0);
    }

    #[test]
    fn test_frame_loader_cache() {
        let config = BatchConfig::minimal();
        let loader = FrameLoader::new(config);

        // Cache should be empty initially
        assert_eq!(loader.cache_len(), 0);

        // Clear should work on empty cache
        loader.clear_cache();
        assert_eq!(loader.cache_len(), 0);
    }
}
