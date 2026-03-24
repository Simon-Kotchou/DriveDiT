//! Configuration structs for DriveDiT data loading.
//!
//! Provides strongly-typed configuration with validation and
//! sensible defaults for all data loading operations.

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{DataError, DataResult};

/// Main dataset configuration for DriveDiT data loading.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    /// Number of frames in each sequence
    #[pyo3(get, set)]
    pub sequence_length: usize,

    /// Target image dimensions (height, width)
    #[pyo3(get, set)]
    pub image_height: usize,
    #[pyo3(get, set)]
    pub image_width: usize,

    /// Number of control signal dimensions (steer, accel, goal_x, goal_y)
    #[pyo3(get, set)]
    pub control_dim: usize,

    /// Temporal stride between frames
    #[pyo3(get, set)]
    pub frame_stride: usize,

    /// Number of context frames for conditioning
    #[pyo3(get, set)]
    pub context_frames: usize,

    /// Whether to load depth data
    #[pyo3(get, set)]
    pub load_depth: bool,

    /// Whether to load flow data
    #[pyo3(get, set)]
    pub load_flow: bool,

    /// Whether to normalize images to [0, 1]
    #[pyo3(get, set)]
    pub normalize_images: bool,

    /// Whether to use fp16 for images
    #[pyo3(get, set)]
    pub use_fp16: bool,

    /// Random seed for reproducibility
    #[pyo3(get, set)]
    pub seed: u64,

    /// Data augmentation probability
    #[pyo3(get, set)]
    pub augment_prob: f32,
}

#[pymethods]
impl DatasetConfig {
    /// Create a new DatasetConfig with specified parameters.
    #[new]
    #[pyo3(signature = (
        sequence_length = 16,
        image_height = 256,
        image_width = 512,
        control_dim = 4,
        frame_stride = 1,
        context_frames = 4,
        load_depth = false,
        load_flow = false,
        normalize_images = true,
        use_fp16 = true,
        seed = 42,
        augment_prob = 0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sequence_length: usize,
        image_height: usize,
        image_width: usize,
        control_dim: usize,
        frame_stride: usize,
        context_frames: usize,
        load_depth: bool,
        load_flow: bool,
        normalize_images: bool,
        use_fp16: bool,
        seed: u64,
        augment_prob: f32,
    ) -> Self {
        Self {
            sequence_length,
            image_height,
            image_width,
            control_dim,
            frame_stride,
            context_frames,
            load_depth,
            load_flow,
            normalize_images,
            use_fp16,
            seed,
            augment_prob,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> PyResult<()> {
        self.validate_inner().map_err(|e| e.into())
    }

    /// Get total frames needed per sample (context + sequence)
    pub fn total_frames(&self) -> usize {
        self.context_frames + self.sequence_length
    }

    /// Create a minimal config for testing
    #[staticmethod]
    pub fn minimal() -> Self {
        Self {
            sequence_length: 4,
            image_height: 64,
            image_width: 128,
            control_dim: 4,
            frame_stride: 1,
            context_frames: 2,
            load_depth: false,
            load_flow: false,
            normalize_images: true,
            use_fp16: false,
            seed: 42,
            augment_prob: 0.0,
        }
    }

    /// Create a production config
    #[staticmethod]
    pub fn production() -> Self {
        Self {
            sequence_length: 16,
            image_height: 256,
            image_width: 512,
            control_dim: 4,
            frame_stride: 1,
            context_frames: 4,
            load_depth: true,
            load_flow: true,
            normalize_images: true,
            use_fp16: true,
            seed: 42,
            augment_prob: 0.2,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetConfig(sequence_length={}, image_size=({}, {}), control_dim={})",
            self.sequence_length, self.image_height, self.image_width, self.control_dim
        )
    }
}

impl DatasetConfig {
    /// Internal validation logic
    fn validate_inner(&self) -> DataResult<()> {
        if self.sequence_length == 0 {
            return Err(DataError::Config(
                "sequence_length must be > 0".to_string(),
            ));
        }
        if self.image_height == 0 || self.image_width == 0 {
            return Err(DataError::Config(
                "image dimensions must be > 0".to_string(),
            ));
        }
        if self.image_height % 8 != 0 || self.image_width % 8 != 0 {
            return Err(DataError::Config(
                "image dimensions must be divisible by 8 for VAE".to_string(),
            ));
        }
        if self.control_dim == 0 {
            return Err(DataError::Config("control_dim must be > 0".to_string()));
        }
        if self.frame_stride == 0 {
            return Err(DataError::Config("frame_stride must be > 0".to_string()));
        }
        if self.augment_prob < 0.0 || self.augment_prob > 1.0 {
            return Err(DataError::Config(
                "augment_prob must be in [0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self::new(16, 256, 512, 4, 1, 4, false, false, true, true, 42, 0.0)
    }
}

/// Telemetry parsing configuration.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Column name for timestamp
    #[pyo3(get, set)]
    pub timestamp_col: String,

    /// Column name for steering angle
    #[pyo3(get, set)]
    pub steering_col: String,

    /// Column name for acceleration/throttle
    #[pyo3(get, set)]
    pub accel_col: String,

    /// Column name for brake
    #[pyo3(get, set)]
    pub brake_col: String,

    /// Column name for speed
    #[pyo3(get, set)]
    pub speed_col: String,

    /// Column names for GPS coordinates (lat, lon)
    #[pyo3(get, set)]
    pub gps_lat_col: String,
    #[pyo3(get, set)]
    pub gps_lon_col: String,

    /// Steering normalization range (min, max)
    #[pyo3(get, set)]
    pub steering_range: (f32, f32),

    /// Speed normalization range (min, max in m/s)
    #[pyo3(get, set)]
    pub speed_range: (f32, f32),

    /// Whether to interpolate missing values
    #[pyo3(get, set)]
    pub interpolate_missing: bool,

    /// Maximum gap (in ms) for interpolation
    #[pyo3(get, set)]
    pub max_interpolation_gap_ms: u64,

    /// Sampling rate in Hz
    #[pyo3(get, set)]
    pub sampling_rate_hz: f32,
}

#[pymethods]
impl TelemetryConfig {
    /// Create a new TelemetryConfig with comma.ai style defaults.
    #[new]
    #[pyo3(signature = (
        timestamp_col = "timestamp".to_string(),
        steering_col = "steering_angle".to_string(),
        accel_col = "accel".to_string(),
        brake_col = "brake".to_string(),
        speed_col = "speed".to_string(),
        gps_lat_col = "latitude".to_string(),
        gps_lon_col = "longitude".to_string(),
        steering_range = (-1.0, 1.0),
        speed_range = (0.0, 40.0),
        interpolate_missing = true,
        max_interpolation_gap_ms = 100,
        sampling_rate_hz = 20.0
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        timestamp_col: String,
        steering_col: String,
        accel_col: String,
        brake_col: String,
        speed_col: String,
        gps_lat_col: String,
        gps_lon_col: String,
        steering_range: (f32, f32),
        speed_range: (f32, f32),
        interpolate_missing: bool,
        max_interpolation_gap_ms: u64,
        sampling_rate_hz: f32,
    ) -> Self {
        Self {
            timestamp_col,
            steering_col,
            accel_col,
            brake_col,
            speed_col,
            gps_lat_col,
            gps_lon_col,
            steering_range,
            speed_range,
            interpolate_missing,
            max_interpolation_gap_ms,
            sampling_rate_hz,
        }
    }

    /// Create config for comma.ai openpilot format
    #[staticmethod]
    pub fn comma_ai() -> Self {
        Self {
            timestamp_col: "t".to_string(),
            steering_col: "steeringAngleDeg".to_string(),
            accel_col: "aEgo".to_string(),
            brake_col: "brakePressed".to_string(),
            speed_col: "vEgo".to_string(),
            gps_lat_col: "latitude".to_string(),
            gps_lon_col: "longitude".to_string(),
            steering_range: (-500.0, 500.0), // degrees
            speed_range: (0.0, 45.0),        // m/s
            interpolate_missing: true,
            max_interpolation_gap_ms: 100,
            sampling_rate_hz: 20.0,
        }
    }

    /// Create config for nuScenes format
    #[staticmethod]
    pub fn nuscenes() -> Self {
        Self {
            timestamp_col: "timestamp".to_string(),
            steering_col: "steering".to_string(),
            accel_col: "acceleration".to_string(),
            brake_col: "brake".to_string(),
            speed_col: "speed".to_string(),
            gps_lat_col: "lat".to_string(),
            gps_lon_col: "lon".to_string(),
            steering_range: (-1.0, 1.0),
            speed_range: (0.0, 30.0),
            interpolate_missing: true,
            max_interpolation_gap_ms: 50,
            sampling_rate_hz: 12.0,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TelemetryConfig(steering_col='{}', speed_col='{}', rate={}Hz)",
            self.steering_col, self.speed_col, self.sampling_rate_hz
        )
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self::comma_ai()
    }
}

/// Data loader configuration for parallel processing.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoaderConfig {
    /// Number of worker threads
    #[pyo3(get, set)]
    pub num_workers: usize,

    /// Number of batches to prefetch
    #[pyo3(get, set)]
    pub prefetch_factor: usize,

    /// Batch size
    #[pyo3(get, set)]
    pub batch_size: usize,

    /// Whether to shuffle data
    #[pyo3(get, set)]
    pub shuffle: bool,

    /// Whether to drop incomplete last batch
    #[pyo3(get, set)]
    pub drop_last: bool,

    /// Pin memory for faster GPU transfer
    #[pyo3(get, set)]
    pub pin_memory: bool,

    /// Maximum memory budget in bytes (0 = unlimited)
    #[pyo3(get, set)]
    pub memory_budget_bytes: usize,

    /// Use memory-mapped files
    #[pyo3(get, set)]
    pub use_mmap: bool,

    /// Cache decoded images in memory
    #[pyo3(get, set)]
    pub cache_decoded: bool,

    /// Maximum cache size in MB
    #[pyo3(get, set)]
    pub cache_size_mb: usize,
}

#[pymethods]
impl LoaderConfig {
    /// Create a new LoaderConfig with specified parameters.
    #[new]
    #[pyo3(signature = (
        num_workers = 4,
        prefetch_factor = 2,
        batch_size = 8,
        shuffle = true,
        drop_last = true,
        pin_memory = true,
        memory_budget_bytes = 0,
        use_mmap = true,
        cache_decoded = false,
        cache_size_mb = 1024
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_workers: usize,
        prefetch_factor: usize,
        batch_size: usize,
        shuffle: bool,
        drop_last: bool,
        pin_memory: bool,
        memory_budget_bytes: usize,
        use_mmap: bool,
        cache_decoded: bool,
        cache_size_mb: usize,
    ) -> Self {
        Self {
            num_workers,
            prefetch_factor,
            batch_size,
            shuffle,
            drop_last,
            pin_memory,
            memory_budget_bytes,
            use_mmap,
            cache_decoded,
            cache_size_mb,
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> PyResult<()> {
        self.validate_inner().map_err(|e| e.into())
    }

    /// Get optimal config based on available system resources
    #[staticmethod]
    pub fn auto_detect() -> Self {
        let num_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        // Use half of available CPUs for data loading
        let num_workers = (num_cpus / 2).max(1).min(16);

        Self {
            num_workers,
            prefetch_factor: 2,
            batch_size: 8,
            shuffle: true,
            drop_last: true,
            pin_memory: true,
            memory_budget_bytes: 0,
            use_mmap: true,
            cache_decoded: false,
            cache_size_mb: 2048,
        }
    }

    /// Create config for inference (minimal latency)
    #[staticmethod]
    pub fn inference() -> Self {
        Self {
            num_workers: 2,
            prefetch_factor: 1,
            batch_size: 1,
            shuffle: false,
            drop_last: false,
            pin_memory: true,
            memory_budget_bytes: 0,
            use_mmap: true,
            cache_decoded: true,
            cache_size_mb: 512,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LoaderConfig(num_workers={}, batch_size={}, prefetch={})",
            self.num_workers, self.batch_size, self.prefetch_factor
        )
    }
}

impl LoaderConfig {
    fn validate_inner(&self) -> DataResult<()> {
        if self.num_workers == 0 {
            return Err(DataError::Config("num_workers must be > 0".to_string()));
        }
        if self.batch_size == 0 {
            return Err(DataError::Config("batch_size must be > 0".to_string()));
        }
        if self.prefetch_factor == 0 {
            return Err(DataError::Config("prefetch_factor must be > 0".to_string()));
        }
        Ok(())
    }
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// ENFCAP file format configuration.
#[pyclass]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnfcapConfig {
    /// Expected magic bytes for validation
    #[pyo3(get, set)]
    pub magic_bytes: Vec<u8>,

    /// Version number expected
    #[pyo3(get, set)]
    pub version: u32,

    /// Whether frames are compressed
    #[pyo3(get, set)]
    pub compressed: bool,

    /// Compression type (0=none, 1=lz4, 2=zstd)
    #[pyo3(get, set)]
    pub compression_type: u8,

    /// Frame width
    #[pyo3(get, set)]
    pub frame_width: u32,

    /// Frame height
    #[pyo3(get, set)]
    pub frame_height: u32,

    /// Frames per second
    #[pyo3(get, set)]
    pub fps: f32,

    /// Bits per channel
    #[pyo3(get, set)]
    pub bits_per_channel: u8,
}

#[pymethods]
impl EnfcapConfig {
    #[new]
    #[pyo3(signature = (
        magic_bytes = vec![0x45, 0x4E, 0x46, 0x43],
        version = 1,
        compressed = true,
        compression_type = 1,
        frame_width = 1920,
        frame_height = 1080,
        fps = 30.0,
        bits_per_channel = 8
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        magic_bytes: Vec<u8>,
        version: u32,
        compressed: bool,
        compression_type: u8,
        frame_width: u32,
        frame_height: u32,
        fps: f32,
        bits_per_channel: u8,
    ) -> Self {
        Self {
            magic_bytes,
            version,
            compressed,
            compression_type,
            frame_width,
            frame_height,
            fps,
            bits_per_channel,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "EnfcapConfig({}x{} @ {}fps, compressed={})",
            self.frame_width, self.frame_height, self.fps, self.compressed
        )
    }
}

impl Default for EnfcapConfig {
    fn default() -> Self {
        Self::new(
            vec![0x45, 0x4E, 0x46, 0x43],
            1,
            true,
            1,
            1920,
            1080,
            30.0,
            8,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_config_validation() {
        let config = DatasetConfig::default();
        assert!(config.validate_inner().is_ok());

        let mut invalid = config.clone();
        invalid.sequence_length = 0;
        assert!(invalid.validate_inner().is_err());
    }

    #[test]
    fn test_dataset_config_total_frames() {
        let config = DatasetConfig::default();
        assert_eq!(config.total_frames(), config.context_frames + config.sequence_length);
    }

    #[test]
    fn test_loader_config_auto_detect() {
        let config = LoaderConfig::auto_detect();
        assert!(config.num_workers > 0);
        assert!(config.validate_inner().is_ok());
    }

    #[test]
    fn test_telemetry_config_presets() {
        let comma = TelemetryConfig::comma_ai();
        assert_eq!(comma.sampling_rate_hz, 20.0);

        let nuscenes = TelemetryConfig::nuscenes();
        assert_eq!(nuscenes.sampling_rate_hz, 12.0);
    }
}
