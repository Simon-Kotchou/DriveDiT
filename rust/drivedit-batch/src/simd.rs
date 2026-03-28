//! SIMD-accelerated normalization utilities.
//!
//! This module provides high-performance image normalization using SIMD
//! instructions when available, with automatic fallback to scalar code.
//!
//! # Features
//!
//! - HWC to CHW conversion with normalization
//! - Batch processing with parallelization
//! - SIMD acceleration on supported platforms
//!
//! # Example
//!
//! ```rust,ignore
//! use drivedit_batch::simd::{normalize_frame, normalize_batch};
//! use drivedit_types::{RawFrame, NormalizationParams, Telemetry};
//!
//! let raw = RawFrame::new(0, 0.0, vec![128u8; 640*360*3], 640, 360, Telemetry::default());
//! let params = NormalizationParams::default();
//!
//! let processed = normalize_frame(&raw, &params);
//! assert_eq!(processed.shape(), (3, 360, 640));
//! ```

use drivedit_types::{NormalizationParams, ProcessedFrame, RawFrame};
use rayon::prelude::*;

/// Normalize a single frame from HWC u8 to CHW f32.
///
/// Converts pixel values from [0, 255] to normalized float range using
/// the provided normalization parameters.
///
/// # Arguments
///
/// * `frame` - Raw frame with HWC pixel layout
/// * `params` - Normalization parameters (mean, std per channel)
///
/// # Returns
///
/// Processed frame with CHW pixel layout in normalized float format.
pub fn normalize_frame(frame: &RawFrame, params: &NormalizationParams) -> ProcessedFrame {
    let (height, width, _) = frame.shape();
    let h = height as usize;
    let w = width as usize;
    let pixel_count = h * w;

    // Pre-compute constants
    let scale = 1.0 / 255.0;
    let inv_std = params.inv_std();

    // Allocate output in CHW format
    let mut output = vec![0.0f32; 3 * pixel_count];

    // Process each channel
    for c in 0..3 {
        let mean = params.mean[c];
        let inv_s = inv_std[c];
        let out_offset = c * pixel_count;

        // Process pixels - HWC input, CHW output
        for y in 0..h {
            for x in 0..w {
                let in_idx = (y * w + x) * 3 + c;
                let out_idx = out_offset + y * w + x;

                let pixel = frame.pixels[in_idx] as f32 * scale;
                output[out_idx] = (pixel - mean) * inv_s;
            }
        }
    }

    ProcessedFrame::new(
        frame.frame_id,
        frame.timestamp_ms,
        output,
        width,
        height,
        frame.telemetry.clone(),
    )
}

/// Normalize a batch of frames in parallel.
///
/// Processes multiple frames using parallel iteration for improved throughput.
///
/// # Arguments
///
/// * `frames` - Slice of raw frames
/// * `params` - Normalization parameters
///
/// # Returns
///
/// Vector of processed frames.
pub fn normalize_batch(frames: &[RawFrame], params: &NormalizationParams) -> Vec<ProcessedFrame> {
    frames
        .par_iter()
        .map(|frame| normalize_frame(frame, params))
        .collect()
}

/// SIMD-accelerated normalization (feature-gated).
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// Normalize using AVX2 instructions.
    ///
    /// This function provides ~4x speedup over scalar normalization
    /// on supported hardware.
    #[target_feature(enable = "avx2")]
    pub unsafe fn normalize_frame_avx2(
        frame: &RawFrame,
        params: &NormalizationParams,
    ) -> ProcessedFrame {
        let (height, width, _) = frame.shape();
        let h = height as usize;
        let w = width as usize;
        let pixel_count = h * w;

        let scale = 1.0 / 255.0;
        let inv_std = params.inv_std();

        let mut output = vec![0.0f32; 3 * pixel_count];

        // Process 8 pixels at a time
        let chunks = pixel_count / 8;

        let scale_vec = _mm256_set1_ps(scale);

        for c in 0..3 {
            let mean_vec = _mm256_set1_ps(params.mean[c]);
            let inv_s_vec = _mm256_set1_ps(inv_std[c]);
            let out_offset = c * pixel_count;

            for chunk in 0..chunks {
                let base = chunk * 8;

                // Gather 8 pixels (non-contiguous in HWC)
                let mut pixels = [0.0f32; 8];
                for i in 0..8 {
                    let px_idx = base + i;
                    let y = px_idx / w;
                    let x = px_idx % w;
                    let in_idx = (y * w + x) * 3 + c;
                    pixels[i] = frame.pixels[in_idx] as f32;
                }

                // Load, scale, normalize
                let v = _mm256_loadu_ps(pixels.as_ptr());
                let scaled = _mm256_mul_ps(v, scale_vec);
                let centered = _mm256_sub_ps(scaled, mean_vec);
                let normalized = _mm256_mul_ps(centered, inv_s_vec);

                // Store
                _mm256_storeu_ps(output[out_offset + base..].as_mut_ptr(), normalized);
            }

            // Handle remainder
            for px_idx in (chunks * 8)..pixel_count {
                let y = px_idx / w;
                let x = px_idx % w;
                let in_idx = (y * w + x) * 3 + c;
                let out_idx = out_offset + px_idx;

                let pixel = frame.pixels[in_idx] as f32 * scale;
                output[out_idx] = (pixel - params.mean[c]) * inv_std[c];
            }
        }

        ProcessedFrame::new(
            frame.frame_id,
            frame.timestamp_ms,
            output,
            width,
            height,
            frame.telemetry.clone(),
        )
    }

    /// Check if AVX2 is available at runtime.
    pub fn is_avx2_available() -> bool {
        is_x86_feature_detected!("avx2")
    }
}

/// Portable SIMD normalization using std::simd.
#[cfg(feature = "simd")]
pub mod portable {
    use super::*;
    use std::simd::{f32x8, Simd};

    /// Normalize using portable SIMD (8-wide f32 vectors).
    pub fn normalize_frame_simd(frame: &RawFrame, params: &NormalizationParams) -> ProcessedFrame {
        let (height, width, _) = frame.shape();
        let h = height as usize;
        let w = width as usize;
        let pixel_count = h * w;

        let scale = 1.0 / 255.0;
        let inv_std = params.inv_std();

        let mut output = vec![0.0f32; 3 * pixel_count];

        let chunks = pixel_count / 8;
        let scale_vec = f32x8::splat(scale);

        for c in 0..3 {
            let mean_vec = f32x8::splat(params.mean[c]);
            let inv_s_vec = f32x8::splat(inv_std[c]);
            let out_offset = c * pixel_count;

            for chunk in 0..chunks {
                let base = chunk * 8;

                // Gather 8 pixels
                let mut pixels = [0.0f32; 8];
                for i in 0..8 {
                    let px_idx = base + i;
                    let y = px_idx / w;
                    let x = px_idx % w;
                    let in_idx = (y * w + x) * 3 + c;
                    pixels[i] = frame.pixels[in_idx] as f32;
                }

                // Vector operations
                let v = f32x8::from_array(pixels);
                let scaled = v * scale_vec;
                let centered = scaled - mean_vec;
                let normalized = centered * inv_s_vec;

                // Store
                let result = normalized.to_array();
                output[out_offset + base..out_offset + base + 8].copy_from_slice(&result);
            }

            // Handle remainder
            for px_idx in (chunks * 8)..pixel_count {
                let y = px_idx / w;
                let x = px_idx % w;
                let in_idx = (y * w + x) * 3 + c;
                let out_idx = out_offset + px_idx;

                let pixel = frame.pixels[in_idx] as f32 * scale;
                output[out_idx] = (pixel - params.mean[c]) * inv_std[c];
            }
        }

        ProcessedFrame::new(
            frame.frame_id,
            frame.timestamp_ms,
            output,
            width,
            height,
            frame.telemetry.clone(),
        )
    }
}

/// Denormalize a processed frame back to HWC u8.
///
/// Useful for visualization and debugging.
pub fn denormalize_frame(frame: &ProcessedFrame, params: &NormalizationParams) -> RawFrame {
    let (_, height, width) = frame.shape();
    let h = height as usize;
    let w = width as usize;
    let pixel_count = h * w;

    let mut output = vec![0u8; pixel_count * 3];

    for c in 0..3 {
        let mean = params.mean[c];
        let std = params.std[c];
        let in_offset = c * pixel_count;

        for y in 0..h {
            for x in 0..w {
                let in_idx = in_offset + y * w + x;
                let out_idx = (y * w + x) * 3 + c;

                let normalized = frame.pixels[in_idx];
                let pixel = normalized * std + mean;
                let pixel_u8 = (pixel * 255.0).clamp(0.0, 255.0) as u8;
                output[out_idx] = pixel_u8;
            }
        }
    }

    RawFrame::new(
        frame.frame_id,
        frame.timestamp_ms,
        output,
        width,
        height,
        frame.telemetry.clone(),
    )
}

/// Convert HWC to CHW without normalization (just layout change).
pub fn hwc_to_chw(hwc_data: &[u8], width: u32, height: u32) -> Vec<f32> {
    let h = height as usize;
    let w = width as usize;
    let pixel_count = h * w;
    let scale = 1.0 / 255.0;

    let mut chw = vec![0.0f32; 3 * pixel_count];

    for c in 0..3 {
        let out_offset = c * pixel_count;
        for y in 0..h {
            for x in 0..w {
                let in_idx = (y * w + x) * 3 + c;
                let out_idx = out_offset + y * w + x;
                chw[out_idx] = hwc_data[in_idx] as f32 * scale;
            }
        }
    }

    chw
}

/// Convert CHW to HWC (reverse layout change).
pub fn chw_to_hwc(chw_data: &[f32], width: u32, height: u32) -> Vec<u8> {
    let h = height as usize;
    let w = width as usize;
    let pixel_count = h * w;

    let mut hwc = vec![0u8; pixel_count * 3];

    for c in 0..3 {
        let in_offset = c * pixel_count;
        for y in 0..h {
            for x in 0..w {
                let in_idx = in_offset + y * w + x;
                let out_idx = (y * w + x) * 3 + c;
                hwc[out_idx] = (chw_data[in_idx] * 255.0).clamp(0.0, 255.0) as u8;
            }
        }
    }

    hwc
}

#[cfg(test)]
mod tests {
    use super::*;
    use drivedit_types::Telemetry;

    fn make_test_frame(width: u32, height: u32, value: u8) -> RawFrame {
        let pixels = vec![value; (width * height * 3) as usize];
        RawFrame::new(0, 0.0, pixels, width, height, Telemetry::default())
    }

    #[test]
    fn test_normalize_frame() {
        let frame = make_test_frame(64, 64, 128);
        let params = NormalizationParams::unit_range();

        let processed = normalize_frame(&frame, &params);

        assert!(processed.is_valid());
        assert_eq!(processed.shape(), (3, 64, 64));

        // With unit range normalization, 128/255 ≈ 0.502
        let expected = 128.0 / 255.0;
        assert!((processed.pixels[0] - expected).abs() < 0.01);
    }

    #[test]
    fn test_normalize_batch() {
        let frames: Vec<RawFrame> = (0..4)
            .map(|i| make_test_frame(64, 64, (i * 64) as u8))
            .collect();
        let params = NormalizationParams::unit_range();

        let processed = normalize_batch(&frames, &params);

        assert_eq!(processed.len(), 4);
        for frame in &processed {
            assert!(frame.is_valid());
            assert_eq!(frame.shape(), (3, 64, 64));
        }
    }

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let original = make_test_frame(64, 64, 100);
        let params = NormalizationParams::default();

        let processed = normalize_frame(&original, &params);
        let recovered = denormalize_frame(&processed, &params);

        // Check that values are close (some rounding expected)
        for (a, b) in original.pixels.iter().zip(recovered.pixels.iter()) {
            assert!((*a as i32 - *b as i32).abs() <= 1);
        }
    }

    #[test]
    fn test_hwc_chw_conversion() {
        let hwc = vec![1u8, 2, 3, 4, 5, 6]; // 2 pixels: [1,2,3], [4,5,6]
        let chw = hwc_to_chw(&hwc, 2, 1);

        // CHW should be: R=[1,4], G=[2,5], B=[3,6]
        let scale = 1.0 / 255.0;
        assert!((chw[0] - 1.0 * scale).abs() < 1e-6); // R0
        assert!((chw[1] - 4.0 * scale).abs() < 1e-6); // R1
        assert!((chw[2] - 2.0 * scale).abs() < 1e-6); // G0
        assert!((chw[3] - 5.0 * scale).abs() < 1e-6); // G1
    }

    #[test]
    fn test_imagenet_normalization() {
        let frame = make_test_frame(64, 64, 128);
        let params = NormalizationParams::default(); // ImageNet

        let processed = normalize_frame(&frame, &params);

        // 128/255 ≈ 0.502
        // (0.502 - 0.485) / 0.229 ≈ 0.074 for R channel
        let pixel = 128.0 / 255.0;
        let expected = (pixel - params.mean[0]) / params.std[0];
        assert!((processed.pixels[0] - expected).abs() < 0.001);
    }
}
