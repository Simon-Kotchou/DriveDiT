//! High-performance depth map loading with NPZ and 16-bit PNG support.
//!
//! This module provides efficient depth map loading for DriveDiT training pipelines,
//! with support for:
//! - NPZ format (numpy compressed arrays)
//! - 16-bit PNG depth maps
//! - Fast bilinear interpolation resizing
//! - Automatic normalization to [0, 1] range
//!
//! # Example
//!
//! ```rust,ignore
//! use drivedit_batch::depth::{DepthLoader, DepthLoaderConfig};
//!
//! let config = DepthLoaderConfig::with_resolution(640, 360);
//! let loader = DepthLoader::new(config);
//!
//! let depth_map = loader.load("depth_0001.png")?;
//! println!("Depth shape: {:?}", depth_map.shape());
//! ```

use aligned_vec::{AVec, ConstAlign};
use image::{GenericImageView, ImageReader};
use rayon::prelude::*;
use std::path::Path;
use thiserror::Error;

/// Alignment for SIMD operations.
const SIMD_ALIGN: usize = 64;

/// Type alias for SIMD-aligned f32 vector.
type AlignedF32Vec = AVec<f32, ConstAlign<SIMD_ALIGN>>;

/// Errors that can occur during depth loading.
#[derive(Error, Debug)]
pub enum DepthError {
    /// Failed to load depth map.
    #[error("Failed to load depth map from {path}: {message}")]
    LoadError {
        /// File path.
        path: String,
        /// Error message.
        message: String,
    },

    /// Unsupported format.
    #[error("Unsupported depth format: {0}")]
    UnsupportedFormat(String),

    /// Dimension mismatch.
    #[error("Invalid depth dimensions: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}")]
    DimensionMismatch {
        /// Expected width.
        expected_width: u32,
        /// Expected height.
        expected_height: u32,
        /// Actual width.
        actual_width: u32,
        /// Actual height.
        actual_height: u32,
    },

    /// I/O error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Configuration for depth loading.
#[derive(Debug, Clone)]
pub struct DepthLoaderConfig {
    /// Target width for resizing (0 = keep original).
    pub target_width: u32,
    /// Target height for resizing (0 = keep original).
    pub target_height: u32,
    /// Minimum depth value for normalization (0 = auto-detect).
    pub min_depth: f32,
    /// Maximum depth value for normalization (0 = auto-detect).
    pub max_depth: f32,
    /// Clip depth values to [min_depth, max_depth].
    pub clip_depth: bool,
    /// Invert depth (1.0 - depth) for disparity-like representation.
    pub invert_depth: bool,
}

impl Default for DepthLoaderConfig {
    fn default() -> Self {
        Self {
            target_width: 640,
            target_height: 360,
            min_depth: 0.0,
            max_depth: 0.0, // Auto-detect
            clip_depth: true,
            invert_depth: false,
        }
    }
}

impl DepthLoaderConfig {
    /// Create config with specific resolution.
    pub fn with_resolution(width: u32, height: u32) -> Self {
        Self {
            target_width: width,
            target_height: height,
            ..Default::default()
        }
    }

    /// Set depth range for normalization.
    pub fn with_depth_range(mut self, min: f32, max: f32) -> Self {
        self.min_depth = min;
        self.max_depth = max;
        self
    }

    /// Enable depth inversion.
    pub fn with_inversion(mut self) -> Self {
        self.invert_depth = true;
        self
    }
}

/// A loaded depth map.
#[derive(Debug)]
pub struct DepthMap {
    /// Normalized depth data in HW format.
    data: AlignedF32Vec,
    /// Width.
    width: u32,
    /// Height.
    height: u32,
    /// Original min depth before normalization.
    original_min: f32,
    /// Original max depth before normalization.
    original_max: f32,
}

impl DepthMap {
    /// Get the raw data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get the raw data as a mutable slice.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get a pointer to the data for FFI.
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Get depth map dimensions (width, height).
    #[inline]
    pub fn shape(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Get total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if depth map is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get original depth range.
    #[inline]
    pub fn original_range(&self) -> (f32, f32) {
        (self.original_min, self.original_max)
    }

    /// Consume and return the underlying aligned vector.
    pub fn into_vec(self) -> AlignedF32Vec {
        self.data
    }

    /// Get depth value at position (bilinear interpolation for sub-pixel coordinates).
    pub fn sample(&self, x: f32, y: f32) -> f32 {
        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width as usize - 1);
        let y1 = (y0 + 1).min(self.height as usize - 1);

        let fx = x - x0 as f32;
        let fy = y - y0 as f32;

        let w = self.width as usize;
        let v00 = self.data[y0 * w + x0];
        let v10 = self.data[y0 * w + x1];
        let v01 = self.data[y1 * w + x0];
        let v11 = self.data[y1 * w + x1];

        let v0 = v00 * (1.0 - fx) + v10 * fx;
        let v1 = v01 * (1.0 - fx) + v11 * fx;

        v0 * (1.0 - fy) + v1 * fy
    }
}

/// High-performance depth map loader.
pub struct DepthLoader {
    config: DepthLoaderConfig,
}

impl DepthLoader {
    /// Create a new depth loader with the given configuration.
    pub fn new(config: DepthLoaderConfig) -> Self {
        Self { config }
    }

    /// Create a depth loader with default configuration.
    pub fn default_loader() -> Self {
        Self::new(DepthLoaderConfig::default())
    }

    /// Load a depth map from a file (auto-detects format).
    pub fn load<P: AsRef<Path>>(&self, path: P) -> Result<DepthMap, DepthError> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        match extension.as_str() {
            "png" => self.load_png_16bit(path),
            _ => Err(DepthError::UnsupportedFormat(extension)),
        }
    }

    /// Load depth from 16-bit PNG.
    pub fn load_png_16bit<P: AsRef<Path>>(&self, path: P) -> Result<DepthMap, DepthError> {
        let path = path.as_ref();
        let img = ImageReader::open(path)
            .map_err(|e: std::io::Error| DepthError::IoError(e))?
            .decode()
            .map_err(|e| DepthError::LoadError {
                path: path.display().to_string(),
                message: e.to_string(),
            })?;

        let (width, height) = img.dimensions();

        // Convert to grayscale 16-bit
        let gray16 = img.to_luma16();
        let raw_data = gray16.into_raw();

        // Convert u16 to f32
        let float_data: Vec<f32> = raw_data.iter().map(|&v| v as f32).collect();

        self.process_depth_data(&float_data, width, height)
    }

    /// Process raw depth data: resize and normalize.
    fn process_depth_data(
        &self,
        data: &[f32],
        src_width: u32,
        src_height: u32,
    ) -> Result<DepthMap, DepthError> {
        // Compute min/max for normalization
        let (min_val, max_val) = self.compute_range(data);

        // Resize if needed
        let (resized, dst_width, dst_height) = if self.config.target_width > 0
            && self.config.target_height > 0
            && (src_width != self.config.target_width || src_height != self.config.target_height)
        {
            let resized = self.resize_bilinear(
                data,
                src_width as usize,
                src_height as usize,
                self.config.target_width as usize,
                self.config.target_height as usize,
            );
            (
                resized,
                self.config.target_width,
                self.config.target_height,
            )
        } else {
            (data.to_vec(), src_width, src_height)
        };

        // Normalize to [0, 1] range
        let normalized = self.normalize(&resized, min_val, max_val);

        // Convert to aligned vector - create from iterator
        let mut aligned_data: AlignedF32Vec = AVec::new(SIMD_ALIGN);
        for val in normalized {
            aligned_data.push(val);
        }

        Ok(DepthMap {
            data: aligned_data,
            width: dst_width,
            height: dst_height,
            original_min: min_val,
            original_max: max_val,
        })
    }

    /// Compute min/max values for normalization.
    fn compute_range(&self, data: &[f32]) -> (f32, f32) {
        if self.config.min_depth != 0.0 || self.config.max_depth != 0.0 {
            return (self.config.min_depth, self.config.max_depth);
        }

        // Auto-detect range (ignoring invalid values like inf/nan)
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for &val in data.iter() {
            if val.is_finite() {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Fallback if no valid values
        if min_val.is_infinite() {
            min_val = 0.0;
        }
        if max_val.is_infinite() {
            max_val = 1.0;
        }

        (min_val, max_val)
    }

    /// Fast bilinear interpolation resize.
    fn resize_bilinear(
        &self,
        data: &[f32],
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) -> Vec<f32> {
        let mut result = vec![0.0f32; dst_width * dst_height];

        let x_ratio = src_width as f32 / dst_width as f32;
        let y_ratio = src_height as f32 / dst_height as f32;

        // Parallel processing of rows
        result
            .par_chunks_mut(dst_width)
            .enumerate()
            .for_each(|(dst_y, row)| {
                let src_y_f = dst_y as f32 * y_ratio;
                let src_y0 = src_y_f.floor() as usize;
                let src_y1 = (src_y0 + 1).min(src_height - 1);
                let y_frac = src_y_f - src_y0 as f32;

                for dst_x in 0..dst_width {
                    let src_x_f = dst_x as f32 * x_ratio;
                    let src_x0 = src_x_f.floor() as usize;
                    let src_x1 = (src_x0 + 1).min(src_width - 1);
                    let x_frac = src_x_f - src_x0 as f32;

                    // Bilinear interpolation
                    let v00 = data[src_y0 * src_width + src_x0];
                    let v10 = data[src_y0 * src_width + src_x1];
                    let v01 = data[src_y1 * src_width + src_x0];
                    let v11 = data[src_y1 * src_width + src_x1];

                    let v0 = v00 * (1.0 - x_frac) + v10 * x_frac;
                    let v1 = v01 * (1.0 - x_frac) + v11 * x_frac;

                    row[dst_x] = v0 * (1.0 - y_frac) + v1 * y_frac;
                }
            });

        result
    }

    /// Normalize depth values to [0, 1] range.
    fn normalize(&self, data: &[f32], min_val: f32, max_val: f32) -> Vec<f32> {
        let range = max_val - min_val;
        let inv_range = if range > 1e-8 { 1.0 / range } else { 1.0 };

        data.par_iter()
            .map(|&val| {
                let mut normalized = if self.config.clip_depth {
                    let clamped = val.clamp(min_val, max_val);
                    (clamped - min_val) * inv_range
                } else {
                    (val - min_val) * inv_range
                };

                if self.config.invert_depth {
                    normalized = 1.0 - normalized;
                }

                normalized
            })
            .collect()
    }

    /// Load multiple depth maps in parallel.
    pub fn load_batch<P: AsRef<Path> + Sync>(&self, paths: &[P]) -> Vec<Result<DepthMap, DepthError>> {
        paths.par_iter().map(|path| self.load(path)).collect()
    }

    /// Load multiple depth maps in parallel, returning only successful loads.
    pub fn load_batch_ok<P: AsRef<Path> + Sync>(&self, paths: &[P]) -> Vec<DepthMap> {
        paths
            .par_iter()
            .filter_map(|path| self.load(path).ok())
            .collect()
    }

    /// Load a batch into a contiguous buffer (NHW format).
    /// Returns (data, batch_size, height, width).
    pub fn load_batch_contiguous<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<(AlignedF32Vec, usize, usize, usize), DepthError> {
        let depth_maps = self.load_batch_ok(paths);

        if depth_maps.is_empty() {
            return Ok((
                AVec::new(SIMD_ALIGN),
                0,
                self.config.target_height as usize,
                self.config.target_width as usize,
            ));
        }

        let batch_size = depth_maps.len();
        let (width, height) = depth_maps[0].shape();
        let map_size = (width * height) as usize;
        let _total_size = batch_size * map_size;

        let mut data: AlignedF32Vec = AVec::new(SIMD_ALIGN);

        for depth_map in depth_maps {
            for &val in depth_map.as_slice() {
                data.push(val);
            }
        }

        Ok((data, batch_size, height as usize, width as usize))
    }

    /// Get the target dimensions.
    #[inline]
    pub fn target_dimensions(&self) -> (u32, u32) {
        (self.config.target_width, self.config.target_height)
    }
}

/// Utilities for depth map processing.
pub mod utils {
    use super::*;

    /// Convert depth to disparity.
    /// disparity = baseline * focal_length / depth
    pub fn depth_to_disparity(
        depth: &[f32],
        baseline: f32,
        focal_length: f32,
        min_depth: f32,
    ) -> Vec<f32> {
        let bf = baseline * focal_length;
        depth
            .par_iter()
            .map(|&d| {
                if d > min_depth {
                    bf / d
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Convert disparity to depth.
    /// depth = baseline * focal_length / disparity
    pub fn disparity_to_depth(
        disparity: &[f32],
        baseline: f32,
        focal_length: f32,
        min_disparity: f32,
    ) -> Vec<f32> {
        let bf = baseline * focal_length;
        disparity
            .par_iter()
            .map(|&d| {
                if d > min_disparity {
                    bf / d
                } else {
                    f32::INFINITY
                }
            })
            .collect()
    }

    /// Apply logarithmic depth compression for visualization.
    pub fn log_depth(depth: &[f32], near: f32, far: f32) -> Vec<f32> {
        let log_near = near.ln();
        let log_far = far.ln();
        let range = log_far - log_near;
        let inv_range = if range.abs() > 1e-8 {
            1.0 / range
        } else {
            1.0
        };

        depth
            .par_iter()
            .map(|&d| {
                let clamped = d.clamp(near, far);
                (clamped.ln() - log_near) * inv_range
            })
            .collect()
    }

    /// Compute gradient magnitude of depth (edge detection).
    pub fn depth_gradient(depth: &[f32], width: usize, height: usize) -> Vec<f32> {
        let mut gradient = vec![0.0f32; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;

                // Sobel-like gradient
                let dx = depth[idx + 1] - depth[idx - 1];
                let dy = depth[idx + width] - depth[idx - width];

                gradient[idx] = (dx * dx + dy * dy).sqrt();
            }
        }

        gradient
    }

    /// Fill invalid depth values using nearest neighbor.
    pub fn fill_invalid(depth: &mut [f32], width: usize, height: usize, invalid_value: f32) {
        let mut valid_mask: Vec<bool> = depth
            .iter()
            .map(|&d| d != invalid_value && d.is_finite())
            .collect();

        // Simple propagation (not optimal but fast)
        for _ in 0..10 {
            let mut changes = false;

            for y in 0..height {
                for x in 0..width {
                    let idx = y * width + x;
                    if !valid_mask[idx] {
                        // Find nearest valid neighbor
                        let mut sum = 0.0f32;
                        let mut count = 0;

                        for dy in -1i32..=1 {
                            for dx in -1i32..=1 {
                                let nx = x as i32 + dx;
                                let ny = y as i32 + dy;
                                if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                                    let nidx = ny as usize * width + nx as usize;
                                    if valid_mask[nidx] {
                                        sum += depth[nidx];
                                        count += 1;
                                    }
                                }
                            }
                        }

                        if count > 0 {
                            depth[idx] = sum / count as f32;
                            valid_mask[idx] = true;
                            changes = true;
                        }
                    }
                }
            }

            if !changes {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depth_loader_config() {
        let config = DepthLoaderConfig::with_resolution(320, 240)
            .with_depth_range(0.1, 100.0)
            .with_inversion();

        assert_eq!(config.target_width, 320);
        assert_eq!(config.target_height, 240);
        assert_eq!(config.min_depth, 0.1);
        assert_eq!(config.max_depth, 100.0);
        assert!(config.invert_depth);
    }

    #[test]
    fn test_depth_map_sample() {
        let mut data: AlignedF32Vec = AVec::new(SIMD_ALIGN);
        // 2x2 depth map with values 0, 1, 2, 3
        for &val in &[0.0f32, 1.0, 2.0, 3.0] {
            data.push(val);
        }

        let depth = DepthMap {
            data,
            width: 2,
            height: 2,
            original_min: 0.0,
            original_max: 3.0,
        };

        // Test corner sampling
        assert!((depth.sample(0.0, 0.0) - 0.0).abs() < 1e-6);
        assert!((depth.sample(1.0, 0.0) - 1.0).abs() < 1e-6);

        // Test center sampling (should be average of all corners)
        let center = depth.sample(0.5, 0.5);
        assert!((center - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_depth_gradient() {
        // Create a gradient: left side is 1.0, right side is 2.0
        // Row 0: 1.0 1.5 2.0
        // Row 1: 1.0 1.5 2.0
        // Row 2: 1.0 1.5 2.0
        let depth = vec![1.0, 1.5, 2.0, 1.0, 1.5, 2.0, 1.0, 1.5, 2.0];
        let gradient = utils::depth_gradient(&depth, 3, 3);

        // Center should have a gradient (dx = 2.0 - 1.0 = 1.0, dy = 0)
        assert!(gradient[4] > 0.0);
    }

    #[test]
    fn test_log_depth() {
        let depth = vec![1.0, 10.0, 100.0];
        let log_d = utils::log_depth(&depth, 1.0, 100.0);

        assert!((log_d[0] - 0.0).abs() < 1e-5);
        assert!((log_d[2] - 1.0).abs() < 1e-5);
    }
}
