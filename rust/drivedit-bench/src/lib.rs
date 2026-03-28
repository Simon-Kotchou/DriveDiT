//! DriveDiT Unified Benchmarks
//!
//! This crate provides comprehensive benchmarks for the DriveDiT data pipeline,
//! testing all components from I/O throughput to streaming latency.
//!
//! # Performance Targets
//!
//! From ENGINEERING_PLAN.md:
//!
//! | Metric                      | Target      |
//! |-----------------------------|-------------|
//! | Frame decode latency        | < 1ms       |
//! | Ring buffer push/pop        | < 1us       |
//! | End-to-end latency          | < 5ms       |
//! | Throughput                  | > 100 fps   |
//!
//! # Benchmark Suites
//!
//! ## I/O Throughput (`io_throughput`)
//!
//! Tests raw I/O performance:
//! - BMP decode throughput (various sizes)
//! - JPEG decode throughput
//! - PNG decode throughput
//! - CSV parse throughput
//!
//! ## Batch Loading (`batch_loading`)
//!
//! Tests batch data loading:
//! - Single frame load time
//! - Parallel batch loading (rayon)
//! - SIMD normalization throughput
//! - Session iteration throughput
//!
//! ## Streaming Latency (`streaming_latency`)
//!
//! Tests real-time streaming performance:
//! - Ring buffer push/pop latency
//! - Frame-telemetry sync latency
//! - End-to-end pipeline latency
//! - Backpressure handling
//!
//! # Running Benchmarks
//!
//! ```bash
//! # Run all benchmarks
//! cargo bench --package drivedit-bench
//!
//! # Run specific benchmark suite
//! cargo bench --bench io_throughput
//! cargo bench --bench batch_loading
//! cargo bench --bench streaming_latency
//!
//! # Run specific benchmark
//! cargo bench -- bmp_decode_640x360
//!
//! # Generate HTML reports
//! cargo bench --bench io_throughput -- --save-baseline main
//! ```
//!
//! # Interpreting Results
//!
//! Criterion reports include:
//! - Mean, median, and standard deviation
//! - Throughput in elements/sec or bytes/sec
//! - Comparison with previous runs
//! - HTML reports with graphs
//!
//! Results are saved to `target/criterion/`.

/// Default frame width for benchmarks.
pub const DEFAULT_WIDTH: u32 = 640;

/// Default frame height for benchmarks.
pub const DEFAULT_HEIGHT: u32 = 360;

/// Default buffer capacity for streaming benchmarks.
pub const DEFAULT_BUFFER_CAPACITY: usize = 256;

/// Default batch size for batch benchmarks.
pub const DEFAULT_BATCH_SIZE: usize = 8;

/// Target frame decode latency in microseconds.
pub const TARGET_DECODE_LATENCY_US: u64 = 1000;

/// Target ring buffer operation latency in microseconds.
pub const TARGET_BUFFER_LATENCY_US: u64 = 1;

/// Target end-to-end pipeline latency in milliseconds.
pub const TARGET_PIPELINE_LATENCY_MS: u64 = 5;

/// Target throughput in frames per second.
pub const TARGET_THROUGHPUT_FPS: u64 = 100;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_WIDTH, 640);
        assert_eq!(DEFAULT_HEIGHT, 360);
        assert_eq!(DEFAULT_BUFFER_CAPACITY, 256);
        assert_eq!(DEFAULT_BATCH_SIZE, 8);
    }

    #[test]
    fn test_targets() {
        assert!(TARGET_DECODE_LATENCY_US <= 1000);
        assert!(TARGET_BUFFER_LATENCY_US <= 1);
        assert!(TARGET_PIPELINE_LATENCY_MS <= 5);
        assert!(TARGET_THROUGHPUT_FPS >= 100);
    }
}
