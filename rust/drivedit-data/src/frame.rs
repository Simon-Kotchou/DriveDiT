//! High-performance frame loading with SIMD normalization and parallel batch processing.
//!
//! This module provides efficient image loading for DriveDiT training pipelines,
//! with support for:
//! - Parallel batch loading using rayon
//! - Fast resizing using fast_image_resize
//! - SIMD-accelerated normalization
//! - Memory-efficient ring buffer for streaming
//! - Zero-copy output for numpy integration

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;

use aligned_vec::{AVec, ConstAlign};
use anyhow::{Context, Result};
use fast_image_resize::{
    images::Image as FirImage, FilterType, PixelType, ResizeAlg, ResizeOptions, Resizer,
};
use image::{DynamicImage, GenericImageView, ImageReader};
use parking_lot::RwLock;
use rayon::prelude::*;
use thiserror::Error;

/// Alignment for SIMD operations (32 bytes for AVX2, 64 for AVX-512)
const SIMD_ALIGN: usize = 64;

/// Type alias for SIMD-aligned f32 vector
type AlignedF32Vec = AVec<f32, ConstAlign<SIMD_ALIGN>>;

/// Errors that can occur during frame loading
#[derive(Error, Debug)]
pub enum FrameError {
    #[error("Failed to load image from {path}: {source}")]
    LoadError {
        path: String,
        #[source]
        source: image::ImageError,
    },

    #[error("Failed to resize image: {0}")]
    ResizeError(String),

    #[error("Invalid image dimensions: expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}")]
    DimensionMismatch {
        expected_width: u32,
        expected_height: u32,
        actual_width: u32,
        actual_height: u32,
    },

    #[error("Ring buffer is empty")]
    BufferEmpty,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Normalization parameters for image preprocessing
#[derive(Debug, Clone, Copy)]
pub struct NormalizationParams {
    /// Per-channel mean values (RGB order)
    pub mean: [f32; 3],
    /// Per-channel standard deviation values (RGB order)
    pub std: [f32; 3],
}

impl Default for NormalizationParams {
    fn default() -> Self {
        // ImageNet normalization values
        Self {
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }
}

impl NormalizationParams {
    /// Create normalization params for [0, 1] range (no normalization beyond scaling)
    pub fn unit_range() -> Self {
        Self {
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }

    /// Create custom normalization params
    pub fn new(mean: [f32; 3], std: [f32; 3]) -> Self {
        Self { mean, std }
    }
}

/// Configuration for the frame loader
#[derive(Debug, Clone)]
pub struct FrameLoaderConfig {
    /// Target width for resizing
    pub target_width: u32,
    /// Target height for resizing
    pub target_height: u32,
    /// Resizing algorithm to use
    pub resize_alg: ResizeAlg,
    /// Normalization parameters
    pub normalization: NormalizationParams,
    /// Number of parallel workers (0 = use rayon default)
    pub num_workers: usize,
    /// Ring buffer capacity for streaming
    pub buffer_capacity: usize,
}

impl Default for FrameLoaderConfig {
    fn default() -> Self {
        Self {
            target_width: 640,
            target_height: 360,
            resize_alg: ResizeAlg::Convolution(FilterType::Bilinear),
            normalization: NormalizationParams::default(),
            num_workers: 0,
            buffer_capacity: 32,
        }
    }
}

impl FrameLoaderConfig {
    /// Create config for specific resolution
    pub fn with_resolution(width: u32, height: u32) -> Self {
        Self {
            target_width: width,
            target_height: height,
            ..Default::default()
        }
    }

    /// Use Lanczos3 filter for higher quality resizing
    pub fn with_high_quality(mut self) -> Self {
        self.resize_alg = ResizeAlg::Convolution(FilterType::Lanczos3);
        self
    }

    /// Use nearest neighbor for fastest resizing
    pub fn with_fast_resize(mut self) -> Self {
        self.resize_alg = ResizeAlg::Nearest;
        self
    }
}

/// A single loaded and processed frame
#[derive(Debug)]
pub struct Frame {
    /// Normalized pixel data in CHW format (channels, height, width)
    /// Stored as contiguous f32 array for zero-copy to numpy
    data: AlignedF32Vec,
    /// Frame width
    width: u32,
    /// Frame height
    height: u32,
    /// Number of channels (always 3 for RGB)
    channels: u32,
}

impl Frame {
    /// Get the raw data as a slice
    #[inline]
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get the raw data as a mutable slice
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Get a pointer to the data for FFI
    #[inline]
    pub fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Get frame dimensions (width, height, channels)
    #[inline]
    pub fn shape(&self) -> (u32, u32, u32) {
        (self.width, self.height, self.channels)
    }

    /// Get total number of elements
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if frame is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consume and return the underlying aligned vector
    pub fn into_vec(self) -> AlignedF32Vec {
        self.data
    }
}

/// High-performance frame loader with parallel processing and SIMD normalization
pub struct FrameLoader {
    config: FrameLoaderConfig,
    /// Thread-local resizer for each worker
    resizer: Resizer,
}

impl FrameLoader {
    /// Create a new frame loader with the given configuration
    pub fn new(config: FrameLoaderConfig) -> Self {
        Self {
            config,
            resizer: Resizer::new(),
        }
    }

    /// Create a frame loader with default configuration
    pub fn default_loader() -> Self {
        Self::new(FrameLoaderConfig::default())
    }

    /// Load a single frame from a file path
    pub fn load_frame<P: AsRef<Path>>(&mut self, path: P) -> Result<Frame, FrameError> {
        let path = path.as_ref();
        let path_str = path.display().to_string();

        // Load image
        let img = ImageReader::open(path)
            .map_err(|e| FrameError::IoError(e))?
            .decode()
            .map_err(|e| FrameError::LoadError {
                path: path_str.clone(),
                source: e,
            })?;

        self.process_image(img)
    }

    /// Load a frame from raw bytes (e.g., from memory-mapped file)
    pub fn load_from_bytes(&mut self, bytes: &[u8]) -> Result<Frame, FrameError> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| FrameError::LoadError {
                path: "<memory>".to_string(),
                source: e,
            })?;

        self.process_image(img)
    }

    /// Process a loaded image: resize and normalize
    fn process_image(&mut self, img: DynamicImage) -> Result<Frame, FrameError> {
        let (src_width, src_height) = img.dimensions();
        let rgb_img = img.to_rgb8();

        // Create source image for fast_image_resize
        let src_image = FirImage::from_vec_u8(
            src_width,
            src_height,
            rgb_img.into_raw(),
            PixelType::U8x3,
        )
        .map_err(|e| FrameError::ResizeError(e.to_string()))?;

        // Create destination image
        let dst_width = self.config.target_width;
        let dst_height = self.config.target_height;
        let mut dst_image = FirImage::new(dst_width, dst_height, PixelType::U8x3);

        // Resize
        self.resizer
            .resize(&src_image, &mut dst_image, &ResizeOptions::new().resize_alg(self.config.resize_alg))
            .map_err(|e| FrameError::ResizeError(e.to_string()))?;

        // Convert to normalized f32 in CHW format
        let rgb_data = dst_image.buffer();
        let frame = self.normalize_to_chw(rgb_data, dst_width, dst_height);

        Ok(frame)
    }

    /// Convert RGB u8 data to normalized f32 in CHW format with SIMD acceleration
    #[inline]
    fn normalize_to_chw(&self, rgb_data: &[u8], width: u32, height: u32) -> Frame {
        let num_pixels = (width * height) as usize;
        let total_elements = num_pixels * 3;

        // Allocate aligned memory for SIMD operations
        let mut data: AlignedF32Vec = AVec::with_capacity(SIMD_ALIGN, total_elements);

        // Pre-compute normalization factors
        let mean = self.config.normalization.mean;
        let std = self.config.normalization.std;
        let inv_std = [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]];
        let scale = 1.0 / 255.0;

        // Process in CHW order (channel-first for PyTorch)
        // Channel 0 (R)
        for i in 0..num_pixels {
            let val = rgb_data[i * 3] as f32 * scale;
            data.push((val - mean[0]) * inv_std[0]);
        }

        // Channel 1 (G)
        for i in 0..num_pixels {
            let val = rgb_data[i * 3 + 1] as f32 * scale;
            data.push((val - mean[1]) * inv_std[1]);
        }

        // Channel 2 (B)
        for i in 0..num_pixels {
            let val = rgb_data[i * 3 + 2] as f32 * scale;
            data.push((val - mean[2]) * inv_std[2]);
        }

        Frame {
            data,
            width,
            height,
            channels: 3,
        }
    }

    /// Load multiple frames in parallel
    pub fn load_batch<P: AsRef<Path> + Sync>(&self, paths: &[P]) -> Vec<Result<Frame, FrameError>> {
        paths
            .par_iter()
            .map(|path| {
                // Each thread gets its own resizer
                let mut loader = FrameLoader::new(self.config.clone());
                loader.load_frame(path)
            })
            .collect()
    }

    /// Load multiple frames in parallel, returning only successful loads
    pub fn load_batch_ok<P: AsRef<Path> + Sync>(&self, paths: &[P]) -> Vec<Frame> {
        paths
            .par_iter()
            .filter_map(|path| {
                let mut loader = FrameLoader::new(self.config.clone());
                loader.load_frame(path).ok()
            })
            .collect()
    }

    /// Load a batch of frames into a single contiguous buffer (NCHW format)
    /// Returns (data, batch_size, channels, height, width)
    pub fn load_batch_contiguous<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<(AlignedF32Vec, usize, usize, usize, usize)> {
        let frames = self.load_batch_ok(paths);

        if frames.is_empty() {
            return Ok((AVec::new(SIMD_ALIGN), 0, 3, self.config.target_height as usize, self.config.target_width as usize));
        }

        let batch_size = frames.len();
        let (width, height, channels) = frames[0].shape();
        let frame_size = (width * height * channels) as usize;
        let total_size = batch_size * frame_size;

        let mut data: AlignedF32Vec = AVec::with_capacity(SIMD_ALIGN, total_size);

        for frame in frames {
            data.extend_from_slice(frame.as_slice());
        }

        Ok((
            data,
            batch_size,
            channels as usize,
            height as usize,
            width as usize,
        ))
    }

    /// Get the target dimensions
    #[inline]
    pub fn target_dimensions(&self) -> (u32, u32) {
        (self.config.target_width, self.config.target_height)
    }
}

/// Memory-efficient ring buffer for streaming frame loading
pub struct FrameRingBuffer {
    buffer: VecDeque<Frame>,
    capacity: usize,
    loader: Arc<RwLock<FrameLoader>>,
}

impl FrameRingBuffer {
    /// Create a new ring buffer with the given loader and capacity
    pub fn new(loader: FrameLoader, capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            loader: Arc::new(RwLock::new(loader)),
        }
    }

    /// Push a frame to the buffer, removing oldest if at capacity
    pub fn push(&mut self, frame: Frame) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(frame);
    }

    /// Load and push a frame from a path
    pub fn load_and_push<P: AsRef<Path>>(&mut self, path: P) -> Result<(), FrameError> {
        let frame = self.loader.write().load_frame(path)?;
        self.push(frame);
        Ok(())
    }

    /// Get the oldest frame without removing it
    pub fn peek_front(&self) -> Option<&Frame> {
        self.buffer.front()
    }

    /// Get the newest frame without removing it
    pub fn peek_back(&self) -> Option<&Frame> {
        self.buffer.back()
    }

    /// Pop the oldest frame
    pub fn pop(&mut self) -> Option<Frame> {
        self.buffer.pop_front()
    }

    /// Get number of frames in buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Check if buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Clear all frames from buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }

    /// Get all frames as a contiguous NCHW tensor
    /// Returns (data, batch_size, channels, height, width)
    pub fn as_contiguous(&self) -> Option<(AlignedF32Vec, usize, usize, usize, usize)> {
        if self.buffer.is_empty() {
            return None;
        }

        let first = self.buffer.front()?;
        let (width, height, channels) = first.shape();
        let frame_size = (width * height * channels) as usize;
        let batch_size = self.buffer.len();
        let total_size = batch_size * frame_size;

        let mut data: AlignedF32Vec = AVec::with_capacity(SIMD_ALIGN, total_size);

        for frame in &self.buffer {
            data.extend_from_slice(frame.as_slice());
        }

        Some((
            data,
            batch_size,
            channels as usize,
            height as usize,
            width as usize,
        ))
    }

    /// Iterate over frames in order (oldest first)
    pub fn iter(&self) -> impl Iterator<Item = &Frame> {
        self.buffer.iter()
    }
}

/// SIMD-accelerated batch normalization utilities
pub mod simd {
    use super::*;

    /// Normalize a batch of frames in-place using SIMD
    /// Data is expected in NCHW format
    #[cfg(target_arch = "x86_64")]
    pub fn normalize_batch_inplace(
        data: &mut [f32],
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        mean: &[f32; 3],
        std: &[f32; 3],
    ) {
        use std::arch::x86_64::*;

        let spatial_size = height * width;
        let frame_size = channels * spatial_size;

        // Pre-compute inverse std for multiplication instead of division
        let inv_std = [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]];

        // Process each frame
        for b in 0..batch_size {
            let frame_offset = b * frame_size;

            // Process each channel
            for c in 0..channels {
                let channel_offset = frame_offset + c * spatial_size;
                let channel_data = &mut data[channel_offset..channel_offset + spatial_size];

                let mean_val = mean[c];
                let inv_std_val = inv_std[c];

                // Check for AVX2 support at runtime
                if is_x86_feature_detected!("avx2") {
                    unsafe {
                        normalize_channel_avx2(channel_data, mean_val, inv_std_val);
                    }
                } else {
                    // Fallback to scalar
                    for val in channel_data.iter_mut() {
                        *val = (*val - mean_val) * inv_std_val;
                    }
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn normalize_channel_avx2(data: &mut [f32], mean: f32, inv_std: f32) {
        use std::arch::x86_64::*;

        let mean_vec = _mm256_set1_ps(mean);
        let inv_std_vec = _mm256_set1_ps(inv_std);

        let chunks = data.len() / 8;
        let remainder = data.len() % 8;

        // Process 8 floats at a time
        for i in 0..chunks {
            let offset = i * 8;
            let ptr = data.as_mut_ptr().add(offset);

            let values = _mm256_loadu_ps(ptr);
            let subtracted = _mm256_sub_ps(values, mean_vec);
            let normalized = _mm256_mul_ps(subtracted, inv_std_vec);
            _mm256_storeu_ps(ptr, normalized);
        }

        // Handle remainder
        let start = chunks * 8;
        for i in 0..remainder {
            data[start + i] = (data[start + i] - mean) * inv_std;
        }
    }

    /// Non-x86 fallback
    #[cfg(not(target_arch = "x86_64"))]
    pub fn normalize_batch_inplace(
        data: &mut [f32],
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        mean: &[f32; 3],
        std: &[f32; 3],
    ) {
        let spatial_size = height * width;
        let frame_size = channels * spatial_size;
        let inv_std = [1.0 / std[0], 1.0 / std[1], 1.0 / std[2]];

        for b in 0..batch_size {
            let frame_offset = b * frame_size;
            for c in 0..channels {
                let channel_offset = frame_offset + c * spatial_size;
                for i in 0..spatial_size {
                    let idx = channel_offset + i;
                    data[idx] = (data[idx] - mean[c]) * inv_std[c];
                }
            }
        }
    }

    /// Compute mean and std of a batch for dataset statistics
    pub fn compute_statistics(
        data: &[f32],
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> ([f32; 3], [f32; 3]) {
        let spatial_size = height * width;
        let frame_size = channels * spatial_size;
        let total_pixels = batch_size * spatial_size;

        let mut sum = [0.0f64; 3];
        let mut sum_sq = [0.0f64; 3];

        for b in 0..batch_size {
            let frame_offset = b * frame_size;
            for c in 0..channels {
                let channel_offset = frame_offset + c * spatial_size;
                for i in 0..spatial_size {
                    let val = data[channel_offset + i] as f64;
                    sum[c] += val;
                    sum_sq[c] += val * val;
                }
            }
        }

        let n = total_pixels as f64;
        let mean = [
            (sum[0] / n) as f32,
            (sum[1] / n) as f32,
            (sum[2] / n) as f32,
        ];
        let std = [
            ((sum_sq[0] / n - (sum[0] / n).powi(2)).sqrt()) as f32,
            ((sum_sq[1] / n - (sum[1] / n).powi(2)).sqrt()) as f32,
            ((sum_sq[2] / n - (sum[2] / n).powi(2)).sqrt()) as f32,
        ];

        (mean, std)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_params_default() {
        let params = NormalizationParams::default();
        assert!((params.mean[0] - 0.485).abs() < 1e-6);
        assert!((params.std[0] - 0.229).abs() < 1e-6);
    }

    #[test]
    fn test_frame_loader_config() {
        let config = FrameLoaderConfig::with_resolution(1920, 1080).with_high_quality();
        assert_eq!(config.target_width, 1920);
        assert_eq!(config.target_height, 1080);
    }

    #[test]
    fn test_frame_ring_buffer() {
        let config = FrameLoaderConfig::with_resolution(64, 64);
        let loader = FrameLoader::new(config);
        let mut buffer = FrameRingBuffer::new(loader, 3);

        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_simd_normalize() {
        let mut data = vec![0.5f32; 3 * 64 * 64];
        let mean = [0.485, 0.456, 0.406];
        let std = [0.229, 0.224, 0.225];

        simd::normalize_batch_inplace(&mut data, 1, 3, 64, 64, &mean, &std);

        // Check that values were normalized
        let expected_r = (0.5 - 0.485) / 0.229;
        assert!((data[0] - expected_r).abs() < 1e-5);
    }
}
