//! Unified configuration types for the DriveDiT data pipeline.
//!
//! This module provides configuration structures for:
//! - [`StreamConfig`]: Real-time streaming configuration
//! - [`BatchConfig`]: Batch loading configuration
//! - [`NormalizationParams`]: Image normalization parameters

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for the streaming pipeline.
///
/// Controls buffer sizes, timeouts, and processing parameters for
/// real-time frame streaming from the Enfusion engine.
///
/// # Example
///
/// ```rust
/// use drivedit_types::StreamConfig;
///
/// let config = StreamConfig::default()
///     .with_dimensions(1920, 1080)
///     .with_capacity(512);
///
/// assert_eq!(config.frame_width, 1920);
/// assert_eq!(config.buffer_capacity, 512);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StreamConfig {
    /// Ring buffer capacity (number of frames).
    pub buffer_capacity: usize,

    /// Default frame width.
    pub frame_width: u32,

    /// Default frame height.
    pub frame_height: u32,

    /// Timeout for waiting on frames (milliseconds).
    pub timeout_ms: u64,

    /// Whether to enable backpressure handling.
    ///
    /// When enabled, the producer will block or drop frames
    /// if the consumer cannot keep up.
    pub enable_backpressure: bool,

    /// Batch size for training data preparation.
    pub batch_size: usize,

    /// Polling interval for file system watcher (milliseconds).
    pub poll_interval_ms: u64,

    /// Maximum frames to hold in memory.
    pub max_memory_frames: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            buffer_capacity: 256,
            frame_width: 640,
            frame_height: 360,
            timeout_ms: 5000,
            enable_backpressure: true,
            batch_size: 8,
            poll_interval_ms: 100,
            max_memory_frames: 1024,
        }
    }
}

impl StreamConfig {
    /// Create a new streaming config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config with custom dimensions.
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.frame_width = width;
        self.frame_height = height;
        self
    }

    /// Create a config with custom buffer capacity.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.buffer_capacity = capacity;
        self
    }

    /// Create a config with custom timeout.
    pub fn with_timeout(mut self, timeout_ms: u64) -> Self {
        self.timeout_ms = timeout_ms;
        self
    }

    /// Create a config with custom batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Get the expected frame size in bytes (RGB u8).
    #[inline]
    pub fn frame_size(&self) -> usize {
        (self.frame_width as usize) * (self.frame_height as usize) * 3
    }

    /// Get total buffer memory usage in bytes.
    #[inline]
    pub fn buffer_memory(&self) -> usize {
        self.buffer_capacity * self.frame_size()
    }

    /// Create a minimal config for testing.
    pub fn minimal() -> Self {
        Self {
            buffer_capacity: 16,
            frame_width: 64,
            frame_height: 64,
            timeout_ms: 1000,
            enable_backpressure: false,
            batch_size: 2,
            poll_interval_ms: 50,
            max_memory_frames: 64,
        }
    }

    /// Create a high-performance config.
    pub fn high_performance() -> Self {
        Self {
            buffer_capacity: 512,
            frame_width: 640,
            frame_height: 360,
            timeout_ms: 10000,
            enable_backpressure: true,
            batch_size: 16,
            poll_interval_ms: 50,
            max_memory_frames: 2048,
        }
    }
}

/// Configuration for batch data loading.
///
/// Controls parallel processing, caching, and memory management
/// for training data loading.
///
/// # Example
///
/// ```rust
/// use drivedit_types::BatchConfig;
///
/// let config = BatchConfig::default();
/// assert!(config.num_workers > 0);
/// assert!(config.shuffle);
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BatchConfig {
    /// Number of parallel worker threads.
    pub num_workers: usize,

    /// Number of batches to prefetch.
    pub prefetch_factor: usize,

    /// Batch size.
    pub batch_size: usize,

    /// Whether to shuffle data.
    pub shuffle: bool,

    /// Whether to drop incomplete last batch.
    pub drop_last: bool,

    /// Target image width for resizing.
    pub target_width: u32,

    /// Target image height for resizing.
    pub target_height: u32,

    /// Number of frames per sequence.
    pub sequence_length: usize,

    /// Temporal stride between frames.
    pub frame_stride: usize,

    /// Number of context frames for conditioning.
    pub context_frames: usize,

    /// Whether to load depth data.
    pub load_depth: bool,

    /// Whether to use memory-mapped files.
    pub use_mmap: bool,

    /// Cache size in MB (0 = disabled).
    pub cache_size_mb: usize,

    /// Random seed for reproducibility.
    pub seed: u64,

    /// Data augmentation probability.
    pub augment_prob: f32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        // Auto-detect number of workers
        let num_workers = std::thread::available_parallelism()
            .map(|n| n.get() / 2)
            .unwrap_or(4)
            .max(1)
            .min(16);

        Self {
            num_workers,
            prefetch_factor: 2,
            batch_size: 8,
            shuffle: true,
            drop_last: true,
            target_width: 640,
            target_height: 360,
            sequence_length: 16,
            frame_stride: 1,
            context_frames: 4,
            load_depth: false,
            use_mmap: true,
            cache_size_mb: 1024,
            seed: 42,
            augment_prob: 0.0,
        }
    }
}

impl BatchConfig {
    /// Create a new batch config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a config for inference (minimal latency).
    pub fn inference() -> Self {
        Self {
            num_workers: 2,
            prefetch_factor: 1,
            batch_size: 1,
            shuffle: false,
            drop_last: false,
            target_width: 640,
            target_height: 360,
            sequence_length: 16,
            frame_stride: 1,
            context_frames: 4,
            load_depth: false,
            use_mmap: true,
            cache_size_mb: 512,
            seed: 42,
            augment_prob: 0.0,
        }
    }

    /// Create a minimal config for testing.
    pub fn minimal() -> Self {
        Self {
            num_workers: 1,
            prefetch_factor: 1,
            batch_size: 2,
            shuffle: false,
            drop_last: false,
            target_width: 64,
            target_height: 64,
            sequence_length: 4,
            frame_stride: 1,
            context_frames: 2,
            load_depth: false,
            use_mmap: false,
            cache_size_mb: 0,
            seed: 42,
            augment_prob: 0.0,
        }
    }

    /// Create a production training config.
    pub fn production() -> Self {
        Self {
            num_workers: 8,
            prefetch_factor: 4,
            batch_size: 16,
            shuffle: true,
            drop_last: true,
            target_width: 640,
            target_height: 360,
            sequence_length: 16,
            frame_stride: 1,
            context_frames: 4,
            load_depth: true,
            use_mmap: true,
            cache_size_mb: 4096,
            seed: 42,
            augment_prob: 0.2,
        }
    }

    /// Get total frames needed per sample.
    #[inline]
    pub fn total_frames(&self) -> usize {
        self.context_frames + self.sequence_length
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.batch_size == 0 {
            return Err("batch_size must be > 0".into());
        }
        if self.num_workers == 0 {
            return Err("num_workers must be > 0".into());
        }
        if self.target_width == 0 || self.target_height == 0 {
            return Err("target dimensions must be > 0".into());
        }
        if self.target_width % 8 != 0 || self.target_height % 8 != 0 {
            return Err("target dimensions must be divisible by 8 for VAE".into());
        }
        if self.sequence_length == 0 {
            return Err("sequence_length must be > 0".into());
        }
        if self.frame_stride == 0 {
            return Err("frame_stride must be > 0".into());
        }
        if self.augment_prob < 0.0 || self.augment_prob > 1.0 {
            return Err("augment_prob must be in [0, 1]".into());
        }
        Ok(())
    }
}

/// Image normalization parameters.
///
/// Per-channel mean and standard deviation for normalizing
/// pixel values from [0, 255] to normalized float range.
///
/// # Example
///
/// ```rust
/// use drivedit_types::NormalizationParams;
///
/// // ImageNet normalization
/// let params = NormalizationParams::default();
/// assert!((params.mean[0] - 0.485).abs() < 1e-6);
///
/// // Simple [0, 1] range
/// let unit = NormalizationParams::unit_range();
/// assert_eq!(unit.mean, [0.0, 0.0, 0.0]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct NormalizationParams {
    /// Per-channel mean values (RGB order).
    pub mean: [f32; 3],

    /// Per-channel standard deviation values (RGB order).
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
    /// Create custom normalization params.
    pub fn new(mean: [f32; 3], std: [f32; 3]) -> Self {
        Self { mean, std }
    }

    /// Create params for [0, 1] range (no normalization beyond scaling).
    pub fn unit_range() -> Self {
        Self {
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        }
    }

    /// Create params for [-1, 1] range (centered normalization).
    pub fn centered() -> Self {
        Self {
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
        }
    }

    /// Get inverse standard deviation for multiplication.
    #[inline]
    pub fn inv_std(&self) -> [f32; 3] {
        [1.0 / self.std[0], 1.0 / self.std[1], 1.0 / self.std[2]]
    }

    /// Normalize a single pixel value for channel c.
    #[inline]
    pub fn normalize(&self, value: f32, channel: usize) -> f32 {
        (value - self.mean[channel]) / self.std[channel]
    }

    /// Denormalize a single pixel value for channel c.
    #[inline]
    pub fn denormalize(&self, value: f32, channel: usize) -> f32 {
        value * self.std[channel] + self.mean[channel]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_defaults() {
        let config = StreamConfig::default();
        assert_eq!(config.buffer_capacity, 256);
        assert_eq!(config.frame_width, 640);
        assert_eq!(config.frame_height, 360);
        assert!(config.enable_backpressure);
    }

    #[test]
    fn test_stream_config_builder() {
        let config = StreamConfig::default()
            .with_dimensions(1920, 1080)
            .with_capacity(512)
            .with_timeout(10000);

        assert_eq!(config.frame_width, 1920);
        assert_eq!(config.frame_height, 1080);
        assert_eq!(config.buffer_capacity, 512);
        assert_eq!(config.timeout_ms, 10000);
    }

    #[test]
    fn test_stream_config_frame_size() {
        let config = StreamConfig::default();
        assert_eq!(config.frame_size(), 640 * 360 * 3);
    }

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert!(config.num_workers > 0);
        assert!(config.shuffle);
        assert!(config.drop_last);
    }

    #[test]
    fn test_batch_config_validation() {
        let config = BatchConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid = config.clone();
        invalid.batch_size = 0;
        assert!(invalid.validate().is_err());

        let mut invalid = config.clone();
        invalid.target_width = 100; // Not divisible by 8
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_batch_config_total_frames() {
        let config = BatchConfig::default();
        assert_eq!(config.total_frames(), config.context_frames + config.sequence_length);
    }

    #[test]
    fn test_normalization_params_default() {
        let params = NormalizationParams::default();
        assert!((params.mean[0] - 0.485).abs() < 1e-6);
        assert!((params.std[0] - 0.229).abs() < 1e-6);
    }

    #[test]
    fn test_normalization_params_unit_range() {
        let params = NormalizationParams::unit_range();
        assert_eq!(params.mean, [0.0, 0.0, 0.0]);
        assert_eq!(params.std, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_normalization_roundtrip() {
        let params = NormalizationParams::default();
        let original = 0.6f32;
        let normalized = params.normalize(original, 0);
        let recovered = params.denormalize(normalized, 0);
        assert!((original - recovered).abs() < 1e-6);
    }

    #[test]
    fn test_normalization_inv_std() {
        let params = NormalizationParams::default();
        let inv = params.inv_std();
        assert!((inv[0] - 1.0 / 0.229).abs() < 1e-6);
    }
}
