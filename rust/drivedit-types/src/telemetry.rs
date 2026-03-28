//! Unified telemetry types for the DriveDiT data pipeline.
//!
//! This module provides the [`Telemetry`] struct representing vehicle state
//! synchronized with each captured frame. Includes position, orientation,
//! speed, and control inputs.
//!
//! # Coordinate System
//!
//! Uses the Enfusion engine coordinate system:
//! - X: Right
//! - Y: Up  
//! - Z: Forward
//!
//! # Example
//!
//! ```rust
//! use drivedit_types::{Telemetry, TELEMETRY_ARRAY_SIZE, CONTROL_ARRAY_SIZE};
//!
//! let telemetry = Telemetry {
//!     pos: [100.0, 50.0, 200.0],
//!     speed_kmh: 60.0,
//!     steering: 0.1,
//!     ..Default::default()
//! };
//!
//! // Convert to array for numpy
//! let array = telemetry.to_array();
//! assert_eq!(array.len(), TELEMETRY_ARRAY_SIZE);
//!
//! // Get 6D control tensor
//! let controls = telemetry.to_control_array();
//! assert_eq!(controls.len(), CONTROL_ARRAY_SIZE);
//! ```

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Number of f32 elements in the full telemetry array representation.
///
/// Layout: `[pos(3), forward(3), up(3), right(3), speed, steering, throttle, brake, accel, dist, heading, heading_rate]`
pub const TELEMETRY_ARRAY_SIZE: usize = 20;

/// Number of f32 elements in the 6D control array.
///
/// Layout: `[steering, acceleration, goal_x, goal_y, speed, heading_rate]`
///
/// This matches the DriveDiT model's expected control tensor format.
pub const CONTROL_ARRAY_SIZE: usize = 6;

/// Vehicle telemetry from capture.
///
/// Contains all vehicle state information synchronized with each frame.
/// Values follow the Enfusion engine coordinate system and units.
///
/// # Fields
///
/// - `pos`: World position in meters
/// - `forward`, `up`, `right`: Orientation vectors (unit vectors)
/// - `speed_kmh`: Speed in kilometers per hour
/// - `steering`: Normalized steering angle [-1, 1]
/// - `throttle`, `brake`: Input values [0, 1]
/// - `acceleration`: Computed longitudinal acceleration in m/s^2
/// - `distance_total`: Session cumulative distance in meters
/// - `heading_rad`: Heading angle in radians [0, 2*PI)
/// - `heading_rate`: Yaw rate in rad/s
///
/// # Example
///
/// ```rust
/// use drivedit_types::Telemetry;
///
/// let telemetry = Telemetry {
///     pos: [100.0, 50.0, 200.0],
///     speed_kmh: 60.0,
///     steering: 0.1,
///     throttle: 0.5,
///     ..Default::default()
/// };
///
/// assert!(!telemetry.is_stationary());
/// assert!(!telemetry.is_braking());
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Telemetry {
    /// Position in world coordinates [x, y, z] (meters).
    pub pos: [f32; 3],

    /// Forward direction vector (unit vector).
    pub forward: [f32; 3],

    /// Up direction vector (unit vector).
    pub up: [f32; 3],

    /// Right direction vector (unit vector).
    pub right: [f32; 3],

    /// Vehicle speed in km/h.
    pub speed_kmh: f32,

    /// Steering angle normalized [-1, 1].
    /// Negative = left, positive = right.
    pub steering: f32,

    /// Throttle input [0, 1].
    pub throttle: f32,

    /// Brake input [0, 1].
    pub brake: f32,

    /// Computed acceleration in m/s^2.
    pub acceleration: f32,

    /// Total distance traveled in meters (session cumulative).
    pub distance_total: f32,

    /// Heading angle in radians [0, 2*PI).
    pub heading_rad: f32,

    /// Heading rate (yaw rate) in rad/s.
    pub heading_rate: f32,
}

impl Telemetry {
    /// Create a new telemetry instance with all zeros.
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert to a flat f32 array for numpy/tensor operations.
    ///
    /// Layout: `[pos(3), forward(3), up(3), right(3), speed, steering, throttle, brake, accel, dist, heading, heading_rate]`
    pub fn to_array(&self) -> [f32; TELEMETRY_ARRAY_SIZE] {
        [
            self.pos[0],
            self.pos[1],
            self.pos[2],
            self.forward[0],
            self.forward[1],
            self.forward[2],
            self.up[0],
            self.up[1],
            self.up[2],
            self.right[0],
            self.right[1],
            self.right[2],
            self.speed_kmh,
            self.steering,
            self.throttle,
            self.brake,
            self.acceleration,
            self.distance_total,
            self.heading_rad,
            self.heading_rate,
        ]
    }

    /// Create telemetry from a flat f32 array.
    ///
    /// Returns `None` if the slice length doesn't match [`TELEMETRY_ARRAY_SIZE`].
    pub fn from_array(arr: &[f32]) -> Option<Self> {
        if arr.len() != TELEMETRY_ARRAY_SIZE {
            return None;
        }
        Some(Self {
            pos: [arr[0], arr[1], arr[2]],
            forward: [arr[3], arr[4], arr[5]],
            up: [arr[6], arr[7], arr[8]],
            right: [arr[9], arr[10], arr[11]],
            speed_kmh: arr[12],
            steering: arr[13],
            throttle: arr[14],
            brake: arr[15],
            acceleration: arr[16],
            distance_total: arr[17],
            heading_rad: arr[18],
            heading_rate: arr[19],
        })
    }

    /// Convert to 6D control array for model input.
    ///
    /// Layout: `[steering, acceleration, goal_x, goal_y, speed, heading_rate]`
    ///
    /// This produces the control tensor format expected by the DriveDiT model.
    /// Note that `goal_x` and `goal_y` default to 0.0 as they require
    /// external goal information.
    ///
    /// # Control Signal Normalization
    ///
    /// Per the DriveDiT spec:
    /// - steering: [-1, 1] (normalized)
    /// - acceleration: [-5, 5] m/s^2
    /// - goal_x: [-50, 50] m (relative position)
    /// - goal_y: [-50, 50] m (relative position)
    /// - speed: [0, 40] m/s
    /// - heading_rate: [-1, 1] rad/s
    pub fn to_control_array(&self) -> [f32; CONTROL_ARRAY_SIZE] {
        [
            self.steering,
            self.acceleration,
            0.0, // goal_x - requires external goal info
            0.0, // goal_y - requires external goal info
            self.speed_kmh / 3.6, // Convert to m/s
            self.heading_rate,
        ]
    }

    /// Create telemetry from a 6D control array.
    ///
    /// Only sets the control-relevant fields; other fields remain at defaults.
    pub fn from_control_array(arr: &[f32]) -> Option<Self> {
        if arr.len() != CONTROL_ARRAY_SIZE {
            return None;
        }
        Some(Self {
            steering: arr[0],
            acceleration: arr[1],
            // goal_x and goal_y at arr[2], arr[3] - not stored in telemetry
            speed_kmh: arr[4] * 3.6, // Convert from m/s
            heading_rate: arr[5],
            ..Default::default()
        })
    }

    /// Convert speed from km/h to m/s.
    #[inline]
    pub fn speed_ms(&self) -> f32 {
        self.speed_kmh / 3.6
    }

    /// Check if the vehicle is stationary (speed < 0.5 km/h).
    #[inline]
    pub fn is_stationary(&self) -> bool {
        self.speed_kmh.abs() < 0.5
    }

    /// Check if braking is active (brake > 0.1).
    #[inline]
    pub fn is_braking(&self) -> bool {
        self.brake > 0.1
    }

    /// Check if accelerating (throttle > 0.1 and not braking).
    #[inline]
    pub fn is_accelerating(&self) -> bool {
        self.throttle > 0.1 && self.brake <= 0.1
    }

    /// Get control vector [steering, throttle, brake].
    pub fn control_vector(&self) -> [f32; 3] {
        [self.steering, self.throttle, self.brake]
    }

    /// Get position as slice reference.
    #[inline]
    pub fn position(&self) -> &[f32; 3] {
        &self.pos
    }

    /// Get the forward direction magnitude (should be ~1.0 for unit vector).
    pub fn forward_magnitude(&self) -> f32 {
        (self.forward[0].powi(2) + self.forward[1].powi(2) + self.forward[2].powi(2)).sqrt()
    }

    /// Get total acceleration magnitude from throttle/brake.
    ///
    /// Positive when accelerating, negative when braking.
    pub fn effective_acceleration(&self) -> f32 {
        if self.is_braking() {
            -self.brake * 5.0 // Approximate max braking decel
        } else {
            self.throttle * 3.0 // Approximate max acceleration
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_defaults() {
        let telemetry = Telemetry::default();
        assert_eq!(telemetry.speed_kmh, 0.0);
        assert!(telemetry.is_stationary());
        assert!(!telemetry.is_braking());
        assert!(!telemetry.is_accelerating());
    }

    #[test]
    fn test_telemetry_to_array() {
        let telemetry = Telemetry {
            pos: [1.0, 2.0, 3.0],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            speed_kmh: 60.0,
            steering: 0.5,
            throttle: 0.8,
            brake: 0.0,
            acceleration: 2.0,
            distance_total: 1000.0,
            heading_rad: 1.57,
            heading_rate: 0.1,
        };

        let arr = telemetry.to_array();
        assert_eq!(arr.len(), TELEMETRY_ARRAY_SIZE);
        assert_eq!(arr[0], 1.0); // pos[0]
        assert_eq!(arr[12], 60.0); // speed_kmh
        assert_eq!(arr[13], 0.5); // steering
    }

    #[test]
    fn test_telemetry_array_roundtrip() {
        let original = Telemetry {
            pos: [1.0, 2.0, 3.0],
            forward: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            right: [1.0, 0.0, 0.0],
            speed_kmh: 60.0,
            steering: 0.5,
            throttle: 0.8,
            brake: 0.1,
            acceleration: 2.0,
            distance_total: 1000.0,
            heading_rad: 1.57,
            heading_rate: 0.1,
        };

        let arr = original.to_array();
        let restored = Telemetry::from_array(&arr).unwrap();
        assert_eq!(original, restored);
    }

    #[test]
    fn test_telemetry_from_array_wrong_size() {
        let arr = [0.0f32; 10]; // Wrong size
        assert!(Telemetry::from_array(&arr).is_none());
    }

    #[test]
    fn test_control_array() {
        let telemetry = Telemetry {
            speed_kmh: 72.0, // 20 m/s
            steering: 0.3,
            acceleration: 1.5,
            heading_rate: 0.2,
            ..Default::default()
        };

        let controls = telemetry.to_control_array();
        assert_eq!(controls.len(), CONTROL_ARRAY_SIZE);
        assert_eq!(controls[0], 0.3); // steering
        assert_eq!(controls[1], 1.5); // acceleration
        assert_eq!(controls[2], 0.0); // goal_x (default)
        assert_eq!(controls[3], 0.0); // goal_y (default)
        assert!((controls[4] - 20.0).abs() < 0.001); // speed in m/s
        assert_eq!(controls[5], 0.2); // heading_rate
    }

    #[test]
    fn test_speed_conversion() {
        let telemetry = Telemetry {
            speed_kmh: 36.0,
            ..Default::default()
        };
        assert!((telemetry.speed_ms() - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_is_stationary() {
        let mut telemetry = Telemetry::default();
        assert!(telemetry.is_stationary());

        telemetry.speed_kmh = 0.4;
        assert!(telemetry.is_stationary());

        telemetry.speed_kmh = 0.6;
        assert!(!telemetry.is_stationary());
    }

    #[test]
    fn test_is_braking() {
        let mut telemetry = Telemetry::default();
        assert!(!telemetry.is_braking());

        telemetry.brake = 0.05;
        assert!(!telemetry.is_braking());

        telemetry.brake = 0.2;
        assert!(telemetry.is_braking());
    }

    #[test]
    fn test_is_accelerating() {
        let mut telemetry = Telemetry::default();
        assert!(!telemetry.is_accelerating());

        telemetry.throttle = 0.5;
        assert!(telemetry.is_accelerating());

        telemetry.brake = 0.2;
        assert!(!telemetry.is_accelerating()); // Braking overrides
    }

    #[test]
    fn test_control_vector() {
        let telemetry = Telemetry {
            steering: 0.3,
            throttle: 0.7,
            brake: 0.1,
            ..Default::default()
        };
        let controls = telemetry.control_vector();
        assert_eq!(controls, [0.3, 0.7, 0.1]);
    }
}

/// Control signal structure (matches comma.ai extended format).
///
/// Used for anchor frame detection and control prediction.
/// The struct is aligned for efficient SIMD operations.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[repr(C, align(32))]
pub struct ControlSignal {
    /// Steering angle (normalized to [-1, 1]).
    pub steer: f32,
    /// Acceleration/brake (positive = accel, negative = brake).
    pub accel: f32,
    /// Goal position X (relative to ego).
    pub goal_x: f32,
    /// Goal position Y (relative to ego).
    pub goal_y: f32,
    /// Speed in m/s.
    pub speed: f32,
    /// Heading rate in rad/s.
    pub heading_rate: f32,
    /// Padding for alignment.
    _pad: [f32; 2],
}

impl ControlSignal {
    /// Create a new control signal.
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

    /// Create with all values.
    pub fn with_full(
        steer: f32,
        accel: f32,
        goal_x: f32,
        goal_y: f32,
        speed: f32,
        heading_rate: f32,
    ) -> Self {
        Self {
            steer,
            accel,
            goal_x,
            goal_y,
            speed,
            heading_rate,
            _pad: [0.0; 2],
        }
    }

    /// Create from a slice of values.
    pub fn from_slice(values: &[f32]) -> Self {
        Self {
            steer: values.first().copied().unwrap_or(0.0),
            accel: values.get(1).copied().unwrap_or(0.0),
            goal_x: values.get(2).copied().unwrap_or(0.0),
            goal_y: values.get(3).copied().unwrap_or(0.0),
            speed: values.get(4).copied().unwrap_or(0.0),
            heading_rate: values.get(5).copied().unwrap_or(0.0),
            _pad: [0.0; 2],
        }
    }

    /// Convert to array.
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

    /// Convert to 8-element array (for SIMD).
    #[inline]
    pub fn to_array_8(&self) -> [f32; 8] {
        [
            self.steer,
            self.accel,
            self.goal_x,
            self.goal_y,
            self.speed,
            self.heading_rate,
            0.0,
            0.0,
        ]
    }

    /// Compute squared difference from another signal.
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

    /// Create from telemetry.
    pub fn from_telemetry(t: &Telemetry) -> Self {
        Self {
            steer: t.steering,
            accel: t.throttle - t.brake,
            goal_x: 0.0,
            goal_y: 0.0,
            speed: t.speed_ms(),
            heading_rate: t.heading_rate,
            _pad: [0.0; 2],
        }
    }
}

#[cfg(test)]
mod control_signal_tests {
    use super::*;

    #[test]
    fn test_control_signal_new() {
        let signal = ControlSignal::new(0.5, 0.3, 1.0, 2.0);
        assert_eq!(signal.steer, 0.5);
        assert_eq!(signal.accel, 0.3);
        assert_eq!(signal.goal_x, 1.0);
        assert_eq!(signal.goal_y, 2.0);
    }

    #[test]
    fn test_control_signal_diff() {
        let a = ControlSignal::new(0.0, 0.0, 0.0, 0.0);
        let b = ControlSignal::new(1.0, 0.0, 0.0, 0.0);
        assert_eq!(a.diff_squared(&b), 1.0);
    }

    #[test]
    fn test_control_signal_from_telemetry() {
        let telemetry = Telemetry {
            speed_kmh: 36.0, // 10 m/s
            steering: 0.5,
            throttle: 0.8,
            brake: 0.2,
            heading_rate: 0.1,
            ..Default::default()
        };

        let signal = ControlSignal::from_telemetry(&telemetry);
        assert_eq!(signal.steer, 0.5);
        assert!((signal.accel - 0.6).abs() < 1e-6); // throttle - brake
        assert!((signal.speed - 10.0).abs() < 1e-6);
        assert_eq!(signal.heading_rate, 0.1);
    }
}
