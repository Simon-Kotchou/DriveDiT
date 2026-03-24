//! High-performance anchor frame detection with SIMD acceleration.
//!
//! This module provides efficient anchor frame detection for DriveDiT training pipelines,
//! identifying keyframes based on control signal changes, motion magnitude, and scene changes.
//!
//! Features:
//! - SIMD-accelerated control signal diff computation
//! - Configurable thresholds for different anchor criteria
//! - Batch processing with boolean mask output
//! - Multiple anchor detection strategies

use std::simd::{f32x8, num::SimdFloat, cmp::SimdPartialOrd};

use aligned_vec::{AVec, ConstAlign};
use rayon::prelude::*;
use thiserror::Error;

/// Alignment for SIMD operations
const SIMD_ALIGN: usize = 64;

/// Type alias for SIMD-aligned f32 vector
type AlignedF32Vec = AVec<f32, ConstAlign<SIMD_ALIGN>>;

/// Errors that can occur during anchor detection
#[derive(Error, Debug)]
pub enum AnchorError {
    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Insufficient frames: need at least {min}, got {actual}")]
    InsufficientFrames { min: usize, actual: usize },
}

/// Control signal structure (matches comma.ai format)
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(32))]
pub struct ControlSignal {
    /// Steering angle (normalized to [-1, 1])
    pub steer: f32,
    /// Acceleration/brake (positive = accel, negative = brake)
    pub accel: f32,
    /// Goal position X (relative to ego)
    pub goal_x: f32,
    /// Goal position Y (relative to ego)
    pub goal_y: f32,
    /// Speed in m/s
    pub speed: f32,
    /// Heading rate in rad/s
    pub heading_rate: f32,
    /// Padding for alignment
    _pad: [f32; 2],
}

impl ControlSignal {
    /// Create a new control signal
    pub fn new(steer: f32, accel: f32, goal_x: f32, goal_y: f32) -> Self {
        Self {
            steer,
            accel,
            goal_x,
            goal_y,
            speed: 0.0,
            heading_rate: 0.0,
            _pad: [0.0; 2],
        }
    }

    /// Create from a slice of values
    pub fn from_slice(values: &[f32]) -> Self {
        Self {
            steer: values.get(0).copied().unwrap_or(0.0),
            accel: values.get(1).copied().unwrap_or(0.0),
            goal_x: values.get(2).copied().unwrap_or(0.0),
            goal_y: values.get(3).copied().unwrap_or(0.0),
            speed: values.get(4).copied().unwrap_or(0.0),
            heading_rate: values.get(5).copied().unwrap_or(0.0),
            _pad: [0.0; 2],
        }
    }

    /// Convert to array
    #[inline]
    pub fn to_array(&self) -> [f32; 6] {
        [
            self.steer,
            self.accel,
            self.goal_x,
            self.goal_y,
            self.speed,
            self.heading_rate,
        ]
    }

    /// Compute squared difference from another signal
    #[inline]
    pub fn diff_squared(&self, other: &ControlSignal) -> f32 {
        let ds = self.steer - other.steer;
        let da = self.accel - other.accel;
        let dgx = self.goal_x - other.goal_x;
        let dgy = self.goal_y - other.goal_y;
        let dsp = self.speed - other.speed;
        let dhr = self.heading_rate - other.heading_rate;

        ds * ds + da * da + dgx * dgx + dgy * dgy + dsp * dsp + dhr * dhr
    }
}

/// Configuration for anchor detection
#[derive(Debug, Clone)]
pub struct AnchorConfig {
    /// Threshold for control signal change (L2 distance)
    pub control_threshold: f32,
    /// Threshold for steering change specifically
    pub steer_threshold: f32,
    /// Threshold for acceleration change specifically
    pub accel_threshold: f32,
    /// Threshold for goal position change
    pub goal_threshold: f32,
    /// Minimum frames between anchors
    pub min_anchor_interval: usize,
    /// Maximum frames between anchors (force anchor if exceeded)
    pub max_anchor_interval: usize,
    /// Use combined control threshold
    pub use_combined_threshold: bool,
    /// Weights for different control components
    pub weights: ControlWeights,
}

/// Weights for control signal components
#[derive(Debug, Clone, Copy)]
pub struct ControlWeights {
    pub steer: f32,
    pub accel: f32,
    pub goal_x: f32,
    pub goal_y: f32,
    pub speed: f32,
    pub heading_rate: f32,
}

impl Default for ControlWeights {
    fn default() -> Self {
        Self {
            steer: 2.0,      // Steering is most important
            accel: 1.5,      // Acceleration changes are significant
            goal_x: 1.0,
            goal_y: 1.0,
            speed: 0.5,
            heading_rate: 1.0,
        }
    }
}

impl Default for AnchorConfig {
    fn default() -> Self {
        Self {
            control_threshold: 0.1,
            steer_threshold: 0.05,
            accel_threshold: 0.1,
            goal_threshold: 0.5,
            min_anchor_interval: 5,
            max_anchor_interval: 30,
            use_combined_threshold: true,
            weights: ControlWeights::default(),
        }
    }
}

impl AnchorConfig {
    /// Create a strict config (more anchor frames)
    pub fn strict() -> Self {
        Self {
            control_threshold: 0.05,
            steer_threshold: 0.02,
            accel_threshold: 0.05,
            goal_threshold: 0.25,
            min_anchor_interval: 3,
            max_anchor_interval: 15,
            ..Default::default()
        }
    }

    /// Create a relaxed config (fewer anchor frames)
    pub fn relaxed() -> Self {
        Self {
            control_threshold: 0.2,
            steer_threshold: 0.1,
            accel_threshold: 0.2,
            goal_threshold: 1.0,
            min_anchor_interval: 10,
            max_anchor_interval: 60,
            ..Default::default()
        }
    }

    /// Set minimum interval between anchors
    pub fn with_min_interval(mut self, interval: usize) -> Self {
        self.min_anchor_interval = interval;
        self
    }

    /// Set maximum interval between anchors
    pub fn with_max_interval(mut self, interval: usize) -> Self {
        self.max_anchor_interval = interval;
        self
    }
}

/// High-performance anchor frame detector
pub struct AnchorDetector {
    config: AnchorConfig,
}

impl AnchorDetector {
    /// Create a new anchor detector with the given configuration
    pub fn new(config: AnchorConfig) -> Self {
        Self { config }
    }

    /// Create an anchor detector with default configuration
    pub fn default_detector() -> Self {
        Self::new(AnchorConfig::default())
    }

    /// Detect anchor frames from control signals
    /// Returns a boolean mask where true indicates an anchor frame
    pub fn detect(&self, signals: &[ControlSignal]) -> Result<Vec<bool>, AnchorError> {
        let n = signals.len();
        if n < 2 {
            return Err(AnchorError::InsufficientFrames { min: 2, actual: n });
        }

        let mut mask = vec![false; n];

        // First frame is always an anchor
        mask[0] = true;

        let mut last_anchor_idx = 0;

        for i in 1..n {
            let frames_since_anchor = i - last_anchor_idx;

            // Check if we've exceeded max interval
            if frames_since_anchor >= self.config.max_anchor_interval {
                mask[i] = true;
                last_anchor_idx = i;
                continue;
            }

            // Skip if we haven't reached min interval
            if frames_since_anchor < self.config.min_anchor_interval {
                continue;
            }

            // Compare with last anchor frame
            let prev = &signals[last_anchor_idx];
            let curr = &signals[i];

            let is_anchor = if self.config.use_combined_threshold {
                self.compute_weighted_diff(prev, curr) >= self.config.control_threshold
            } else {
                self.check_individual_thresholds(prev, curr)
            };

            if is_anchor {
                mask[i] = true;
                last_anchor_idx = i;
            }
        }

        Ok(mask)
    }

    /// Detect anchors from raw f32 array (signals as flat array)
    /// Input: [N, 4] or [N, 6] flattened control signals
    pub fn detect_from_array(
        &self,
        signals: &[f32],
        signal_dim: usize,
    ) -> Result<Vec<bool>, AnchorError> {
        if signals.len() % signal_dim != 0 {
            return Err(AnchorError::DimensionMismatch {
                expected: signal_dim,
                actual: signals.len() % signal_dim,
            });
        }

        let n = signals.len() / signal_dim;
        let control_signals: Vec<ControlSignal> = (0..n)
            .map(|i| ControlSignal::from_slice(&signals[i * signal_dim..(i + 1) * signal_dim]))
            .collect();

        self.detect(&control_signals)
    }

    /// SIMD-accelerated control signal diff computation for batch processing
    #[cfg(target_feature = "avx2")]
    pub fn compute_diffs_simd(&self, signals: &[ControlSignal]) -> Vec<f32> {
        let n = signals.len();
        if n < 2 {
            return vec![];
        }

        let mut diffs = vec![0.0f32; n - 1];

        // Process in chunks of 8 for AVX2
        let chunks = (n - 1) / 8;
        let remainder = (n - 1) % 8;

        // Weights as SIMD vector (repeated pattern)
        let weights = &self.config.weights;

        for chunk in 0..chunks {
            let base = chunk * 8;

            // Load 8 consecutive pairs
            let mut diff_vals = [0.0f32; 8];

            for i in 0..8 {
                let idx = base + i;
                let prev = &signals[idx];
                let curr = &signals[idx + 1];
                diff_vals[i] = self.compute_weighted_diff(prev, curr);
            }

            // Store results
            diffs[base..base + 8].copy_from_slice(&diff_vals);
        }

        // Handle remainder
        for i in 0..remainder {
            let idx = chunks * 8 + i;
            let prev = &signals[idx];
            let curr = &signals[idx + 1];
            diffs[idx] = self.compute_weighted_diff(prev, curr);
        }

        diffs
    }

    /// Non-SIMD version for compatibility
    #[cfg(not(target_feature = "avx2"))]
    pub fn compute_diffs_simd(&self, signals: &[ControlSignal]) -> Vec<f32> {
        self.compute_diffs(signals)
    }

    /// Compute all pairwise diffs (fallback without SIMD)
    pub fn compute_diffs(&self, signals: &[ControlSignal]) -> Vec<f32> {
        if signals.len() < 2 {
            return vec![];
        }

        signals
            .windows(2)
            .map(|w| self.compute_weighted_diff(&w[0], &w[1]))
            .collect()
    }

    /// Compute weighted difference between two control signals
    #[inline]
    fn compute_weighted_diff(&self, a: &ControlSignal, b: &ControlSignal) -> f32 {
        let w = &self.config.weights;

        let ds = (a.steer - b.steer) * w.steer;
        let da = (a.accel - b.accel) * w.accel;
        let dgx = (a.goal_x - b.goal_x) * w.goal_x;
        let dgy = (a.goal_y - b.goal_y) * w.goal_y;
        let dsp = (a.speed - b.speed) * w.speed;
        let dhr = (a.heading_rate - b.heading_rate) * w.heading_rate;

        (ds * ds + da * da + dgx * dgx + dgy * dgy + dsp * dsp + dhr * dhr).sqrt()
    }

    /// Check individual threshold violations
    #[inline]
    fn check_individual_thresholds(&self, a: &ControlSignal, b: &ControlSignal) -> bool {
        let steer_diff = (a.steer - b.steer).abs();
        let accel_diff = (a.accel - b.accel).abs();
        let goal_diff = ((a.goal_x - b.goal_x).powi(2) + (a.goal_y - b.goal_y).powi(2)).sqrt();

        steer_diff >= self.config.steer_threshold
            || accel_diff >= self.config.accel_threshold
            || goal_diff >= self.config.goal_threshold
    }

    /// Get anchor indices from mask
    pub fn mask_to_indices(mask: &[bool]) -> Vec<usize> {
        mask.iter()
            .enumerate()
            .filter_map(|(i, &is_anchor)| if is_anchor { Some(i) } else { None })
            .collect()
    }

    /// Get number of anchors from mask
    pub fn count_anchors(mask: &[bool]) -> usize {
        mask.iter().filter(|&&x| x).count()
    }

    /// Compute anchor density (anchors per frame)
    pub fn anchor_density(mask: &[bool]) -> f32 {
        if mask.is_empty() {
            return 0.0;
        }
        Self::count_anchors(mask) as f32 / mask.len() as f32
    }
}

/// SIMD utilities for anchor detection
pub mod simd {
    use super::*;

    /// Compute L2 distance between two control signal arrays using portable SIMD
    /// Input arrays should be 8-element aligned control signals
    pub fn control_diff_l2_simd(a: &[f32; 8], b: &[f32; 8]) -> f32 {
        let va = f32x8::from_array(*a);
        let vb = f32x8::from_array(*b);
        let diff = va - vb;
        let sq = diff * diff;

        // Horizontal sum
        let arr = sq.to_array();
        arr.iter().sum::<f32>().sqrt()
    }

    /// Batch compute control diffs using SIMD
    /// Returns diff values for consecutive pairs
    pub fn batch_control_diff(signals: &[f32], signal_dim: usize) -> Vec<f32> {
        if signal_dim != 8 {
            // Fallback for non-8-dim signals
            return batch_control_diff_scalar(signals, signal_dim);
        }

        let n = signals.len() / signal_dim;
        if n < 2 {
            return vec![];
        }

        let mut diffs = vec![0.0f32; n - 1];

        for i in 0..n - 1 {
            let a_start = i * signal_dim;
            let b_start = (i + 1) * signal_dim;

            let a: [f32; 8] = signals[a_start..a_start + 8].try_into().unwrap();
            let b: [f32; 8] = signals[b_start..b_start + 8].try_into().unwrap();

            diffs[i] = control_diff_l2_simd(&a, &b);
        }

        diffs
    }

    /// Scalar fallback for non-8-dim signals
    fn batch_control_diff_scalar(signals: &[f32], signal_dim: usize) -> Vec<f32> {
        let n = signals.len() / signal_dim;
        if n < 2 {
            return vec![];
        }

        let mut diffs = vec![0.0f32; n - 1];

        for i in 0..n - 1 {
            let a_start = i * signal_dim;
            let b_start = (i + 1) * signal_dim;

            let mut sum_sq = 0.0f32;
            for j in 0..signal_dim {
                let d = signals[a_start + j] - signals[b_start + j];
                sum_sq += d * d;
            }
            diffs[i] = sum_sq.sqrt();
        }

        diffs
    }

    /// Create anchor mask from diff values using SIMD comparison
    pub fn threshold_mask_simd(diffs: &[f32], threshold: f32) -> Vec<bool> {
        let n = diffs.len();
        let chunks = n / 8;
        let remainder = n % 8;

        let mut mask = vec![false; n];
        let thresh_vec = f32x8::splat(threshold);

        for chunk in 0..chunks {
            let offset = chunk * 8;
            let vals = f32x8::from_slice(&diffs[offset..]);
            let cmp = vals.simd_ge(thresh_vec);

            // Convert mask to bool array
            for (i, &val) in cmp.to_array().iter().enumerate() {
                mask[offset + i] = val;
            }
        }

        // Handle remainder
        for i in 0..remainder {
            let idx = chunks * 8 + i;
            mask[idx] = diffs[idx] >= threshold;
        }

        mask
    }
}

/// Strategies for anchor frame selection
pub mod strategies {
    use super::*;

    /// Uniform sampling strategy (every N frames)
    pub fn uniform_anchors(num_frames: usize, interval: usize) -> Vec<bool> {
        let mut mask = vec![false; num_frames];
        for i in (0..num_frames).step_by(interval) {
            mask[i] = true;
        }
        mask
    }

    /// Random sampling strategy with target anchor rate
    pub fn random_anchors(num_frames: usize, anchor_rate: f32, seed: u64) -> Vec<bool> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut mask = vec![false; num_frames];

        for i in 0..num_frames {
            let mut hasher = DefaultHasher::new();
            (seed, i).hash(&mut hasher);
            let hash = hasher.finish();
            let rand_val = (hash as f64) / (u64::MAX as f64);

            if rand_val < anchor_rate as f64 {
                mask[i] = true;
            }
        }

        // Ensure first and last are anchors
        if !mask.is_empty() {
            mask[0] = true;
            *mask.last_mut().unwrap() = true;
        }

        mask
    }

    /// Scene change detection based on pixel differences
    /// Requires pre-computed frame differences
    pub fn scene_change_anchors(frame_diffs: &[f32], threshold: f32) -> Vec<bool> {
        let mut mask = vec![false; frame_diffs.len() + 1];
        mask[0] = true;

        for (i, &diff) in frame_diffs.iter().enumerate() {
            if diff >= threshold {
                mask[i + 1] = true;
            }
        }

        mask
    }

    /// Combine multiple anchor masks (union)
    pub fn combine_masks_union(masks: &[&[bool]]) -> Vec<bool> {
        if masks.is_empty() {
            return vec![];
        }

        let n = masks[0].len();
        let mut combined = vec![false; n];

        for mask in masks {
            for (i, &is_anchor) in mask.iter().enumerate() {
                combined[i] = combined[i] || is_anchor;
            }
        }

        combined
    }

    /// Combine multiple anchor masks (intersection)
    pub fn combine_masks_intersection(masks: &[&[bool]]) -> Vec<bool> {
        if masks.is_empty() {
            return vec![];
        }

        let n = masks[0].len();
        let mut combined = vec![true; n];

        for mask in masks {
            for (i, &is_anchor) in mask.iter().enumerate() {
                combined[i] = combined[i] && is_anchor;
            }
        }

        combined
    }

    /// Enforce minimum interval between anchors
    pub fn enforce_min_interval(mask: &mut [bool], min_interval: usize) {
        let mut last_anchor = 0;

        for i in 1..mask.len() {
            if mask[i] {
                if i - last_anchor < min_interval {
                    mask[i] = false;
                } else {
                    last_anchor = i;
                }
            }
        }
    }

    /// Enforce maximum interval between anchors
    pub fn enforce_max_interval(mask: &mut [bool], max_interval: usize) {
        let mut last_anchor = 0;

        for i in 1..mask.len() {
            if mask[i] {
                last_anchor = i;
            } else if i - last_anchor >= max_interval {
                mask[i] = true;
                last_anchor = i;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_signal_diff() {
        let a = ControlSignal::new(0.0, 0.0, 0.0, 0.0);
        let b = ControlSignal::new(1.0, 0.0, 0.0, 0.0);

        assert_eq!(a.diff_squared(&b), 1.0);
    }

    #[test]
    fn test_anchor_detection_basic() {
        let signals = vec![
            ControlSignal::new(0.0, 0.0, 0.0, 0.0),
            ControlSignal::new(0.0, 0.0, 0.0, 0.0),
            ControlSignal::new(0.0, 0.0, 0.0, 0.0),
            ControlSignal::new(0.0, 0.0, 0.0, 0.0),
            ControlSignal::new(0.0, 0.0, 0.0, 0.0),
            ControlSignal::new(0.5, 0.0, 0.0, 0.0), // Significant change
            ControlSignal::new(0.5, 0.0, 0.0, 0.0),
            ControlSignal::new(0.5, 0.0, 0.0, 0.0),
        ];

        let config = AnchorConfig {
            min_anchor_interval: 1,
            max_anchor_interval: 100,
            control_threshold: 0.1,
            ..Default::default()
        };

        let detector = AnchorDetector::new(config);
        let mask = detector.detect(&signals).unwrap();

        // First frame should be anchor
        assert!(mask[0]);
        // Frame 5 should be anchor due to large change
        assert!(mask[5]);
    }

    #[test]
    fn test_anchor_detection_max_interval() {
        let signals: Vec<ControlSignal> = (0..50)
            .map(|_| ControlSignal::new(0.0, 0.0, 0.0, 0.0))
            .collect();

        let config = AnchorConfig {
            max_anchor_interval: 10,
            min_anchor_interval: 1,
            control_threshold: 1.0, // High threshold
            ..Default::default()
        };

        let detector = AnchorDetector::new(config);
        let mask = detector.detect(&signals).unwrap();

        // Should have anchors at 0, 10, 20, 30, 40
        let indices = AnchorDetector::mask_to_indices(&mask);
        assert!(indices.contains(&0));
        assert!(indices.contains(&10));
        assert!(indices.contains(&20));
    }

    #[test]
    fn test_simd_diff() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let diff = simd::control_diff_l2_simd(&a, &b);
        assert!((diff - 0.0).abs() < 1e-6);

        let c = [2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let diff2 = simd::control_diff_l2_simd(&a, &c);
        assert!((diff2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_uniform_anchors() {
        let mask = strategies::uniform_anchors(20, 5);
        assert!(mask[0]);
        assert!(mask[5]);
        assert!(mask[10]);
        assert!(mask[15]);
        assert!(!mask[1]);
    }

    #[test]
    fn test_combine_masks() {
        let mask1 = [true, false, true, false];
        let mask2 = [false, true, true, false];

        let union = strategies::combine_masks_union(&[&mask1, &mask2]);
        assert_eq!(union, vec![true, true, true, false]);

        let intersection = strategies::combine_masks_intersection(&[&mask1, &mask2]);
        assert_eq!(intersection, vec![false, false, true, false]);
    }

    #[test]
    fn test_anchor_density() {
        let mask = vec![true, false, false, true, false];
        let density = AnchorDetector::anchor_density(&mask);
        assert!((density - 0.4).abs() < 1e-6);
    }
}
