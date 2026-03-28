//! Python type wrappers for DriveDiT data types.
//!
//! Provides PyO3-compatible wrappers around the core Rust types with
//! convenient conversions to numpy arrays.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Size of the telemetry array representation.
pub const TELEMETRY_ARRAY_SIZE: usize = 20;

/// Python wrapper for vehicle telemetry data.
///
/// Contains all vehicle state information synchronized with each frame.
///
/// # Example
///
/// ```python
/// from drivedit import Telemetry
///
/// tel = Telemetry()
/// tel.speed_kmh = 60.0
/// tel.steering = 0.1
///
/// # Convert to numpy array
/// arr = tel.to_array()  # [20] float32
///
/// # Get 6D control vector for training
/// ctrl = tel.to_control_array()  # [6] float32
/// ```
#[pyclass(name = "Telemetry")]
#[derive(Debug, Clone, Default)]
pub struct PyTelemetry {
    /// Position in world coordinates [x, y, z] (meters).
    #[pyo3(get, set)]
    pub pos_x: f32,
    #[pyo3(get, set)]
    pub pos_y: f32,
    #[pyo3(get, set)]
    pub pos_z: f32,

    /// Forward direction vector (unit vector).
    #[pyo3(get, set)]
    pub forward_x: f32,
    #[pyo3(get, set)]
    pub forward_y: f32,
    #[pyo3(get, set)]
    pub forward_z: f32,

    /// Up direction vector (unit vector).
    #[pyo3(get, set)]
    pub up_x: f32,
    #[pyo3(get, set)]
    pub up_y: f32,
    #[pyo3(get, set)]
    pub up_z: f32,

    /// Right direction vector (unit vector).
    #[pyo3(get, set)]
    pub right_x: f32,
    #[pyo3(get, set)]
    pub right_y: f32,
    #[pyo3(get, set)]
    pub right_z: f32,

    /// Vehicle speed in km/h.
    #[pyo3(get, set)]
    pub speed_kmh: f32,

    /// Steering angle normalized [-1, 1].
    #[pyo3(get, set)]
    pub steering: f32,

    /// Throttle input [0, 1].
    #[pyo3(get, set)]
    pub throttle: f32,

    /// Brake input [0, 1].
    #[pyo3(get, set)]
    pub brake: f32,

    /// Computed acceleration in m/s^2.
    #[pyo3(get, set)]
    pub acceleration: f32,

    /// Total distance traveled in meters.
    #[pyo3(get, set)]
    pub distance_total: f32,

    /// Heading angle in radians [0, 2*PI).
    #[pyo3(get, set)]
    pub heading_rad: f32,

    /// Heading rate (yaw rate) in rad/s.
    #[pyo3(get, set)]
    pub heading_rate: f32,
}

#[pymethods]
impl PyTelemetry {
    /// Create a new Telemetry with default values.
    #[new]
    #[pyo3(signature = (
        speed_kmh = 0.0,
        steering = 0.0,
        throttle = 0.0,
        brake = 0.0,
        acceleration = 0.0,
        heading_rate = 0.0
    ))]
    fn new(
        speed_kmh: f32,
        steering: f32,
        throttle: f32,
        brake: f32,
        acceleration: f32,
        heading_rate: f32,
    ) -> Self {
        Self {
            speed_kmh,
            steering,
            throttle,
            brake,
            acceleration,
            heading_rate,
            ..Default::default()
        }
    }

    /// Convert to a flat f32 array [20] for numpy/tensor operations.
    ///
    /// Layout: [pos(3), forward(3), up(3), right(3), speed, steering, throttle, brake, accel, dist, heading, heading_rate]
    fn to_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let arr = [
            self.pos_x,
            self.pos_y,
            self.pos_z,
            self.forward_x,
            self.forward_y,
            self.forward_z,
            self.up_x,
            self.up_y,
            self.up_z,
            self.right_x,
            self.right_y,
            self.right_z,
            self.speed_kmh,
            self.steering,
            self.throttle,
            self.brake,
            self.acceleration,
            self.distance_total,
            self.heading_rad,
            self.heading_rate,
        ];
        arr.to_vec().into_pyarray(py)
    }

    /// Convert to 6D control tensor for training.
    ///
    /// Returns: [steering, accel, goal_x, goal_y, speed, heading_rate]
    fn to_control_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let arr = [
            self.steering,
            self.throttle - self.brake, // Combined acceleration
            0.0,                         // goal_x (filled by planner)
            0.0,                         // goal_y
            self.speed_kmh / 120.0,      // Normalized speed
            self.heading_rate,
        ];
        arr.to_vec().into_pyarray(py)
    }

    /// Create telemetry from a flat array.
    #[staticmethod]
    fn from_array(arr: Vec<f32>) -> PyResult<Self> {
        if arr.len() != TELEMETRY_ARRAY_SIZE {
            return Err(PyValueError::new_err(format!(
                "Expected array of length {}, got {}",
                TELEMETRY_ARRAY_SIZE,
                arr.len()
            )));
        }
        Ok(Self {
            pos_x: arr[0],
            pos_y: arr[1],
            pos_z: arr[2],
            forward_x: arr[3],
            forward_y: arr[4],
            forward_z: arr[5],
            up_x: arr[6],
            up_y: arr[7],
            up_z: arr[8],
            right_x: arr[9],
            right_y: arr[10],
            right_z: arr[11],
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

    /// Get speed in m/s.
    fn speed_ms(&self) -> f32 {
        self.speed_kmh / 3.6
    }

    /// Check if vehicle is stationary (speed < 0.5 km/h).
    fn is_stationary(&self) -> bool {
        self.speed_kmh.abs() < 0.5
    }

    /// Check if braking is active (brake > 0.1).
    fn is_braking(&self) -> bool {
        self.brake > 0.1
    }

    /// Get position as a tuple.
    fn position(&self) -> (f32, f32, f32) {
        (self.pos_x, self.pos_y, self.pos_z)
    }

    /// Get forward direction as a tuple.
    fn forward(&self) -> (f32, f32, f32) {
        (self.forward_x, self.forward_y, self.forward_z)
    }

    fn __repr__(&self) -> String {
        format!(
            "Telemetry(speed={:.1}km/h, steering={:.2}, throttle={:.2}, brake={:.2})",
            self.speed_kmh, self.steering, self.throttle, self.brake
        )
    }
}

/// Python wrapper for a single frame with pixels and telemetry.
///
/// # Example
///
/// ```python
/// from drivedit import Frame
///
/// # Access frame data
/// frame.frame_id      # int
/// frame.timestamp_ms  # float
/// frame.width         # int
/// frame.height        # int
/// frame.pixels        # numpy array [H, W, 3] uint8
/// frame.telemetry     # Telemetry object
/// ```
#[pyclass(name = "Frame")]
#[derive(Debug, Clone)]
pub struct PyFrame {
    /// Frame ID (monotonic counter).
    #[pyo3(get)]
    pub frame_id: u64,

    /// Capture timestamp in milliseconds.
    #[pyo3(get)]
    pub timestamp_ms: f64,

    /// Frame width in pixels.
    #[pyo3(get)]
    pub width: u32,

    /// Frame height in pixels.
    #[pyo3(get)]
    pub height: u32,

    /// RGB pixel data in HWC format.
    pixels: Vec<u8>,

    /// Associated telemetry.
    telemetry: PyTelemetry,
}

#[pymethods]
impl PyFrame {
    /// Create a new Frame.
    #[new]
    #[pyo3(signature = (frame_id, timestamp_ms, pixels, width, height, telemetry = None))]
    fn new(
        frame_id: u64,
        timestamp_ms: f64,
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        telemetry: Option<PyTelemetry>,
    ) -> Self {
        Self {
            frame_id,
            timestamp_ms,
            width,
            height,
            pixels,
            telemetry: telemetry.unwrap_or_default(),
        }
    }

    /// Get pixel data as numpy array [H, W, 3] uint8.
    fn pixels<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<u8>>> {
        let shape = [self.height as usize, self.width as usize, 3];
        let arr = self.pixels.clone().into_pyarray(py);
        arr.reshape(shape)
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Get pixel data as numpy array [C, H, W] float32 (normalized).
    fn pixels_chw<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray3<f32>>> {
        let h = self.height as usize;
        let w = self.width as usize;

        // Convert HWC u8 to CHW f32
        let mut chw = vec![0.0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let src_idx = (y * w + x) * 3;
                chw[0 * h * w + y * w + x] = self.pixels[src_idx] as f32 / 255.0;
                chw[1 * h * w + y * w + x] = self.pixels[src_idx + 1] as f32 / 255.0;
                chw[2 * h * w + y * w + x] = self.pixels[src_idx + 2] as f32 / 255.0;
            }
        }

        let arr = chw.into_pyarray(py);
        arr.reshape([3, h, w])
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Get associated telemetry.
    fn telemetry(&self) -> PyTelemetry {
        self.telemetry.clone()
    }

    /// Check if frame data is valid (correct size).
    fn is_valid(&self) -> bool {
        let expected = (self.width as usize) * (self.height as usize) * 3;
        self.pixels.len() == expected
    }

    /// Get expected pixel buffer size.
    fn expected_size(&self) -> usize {
        (self.width as usize) * (self.height as usize) * 3
    }

    fn __repr__(&self) -> String {
        format!(
            "Frame(id={}, {}x{}, ts={:.1}ms)",
            self.frame_id, self.width, self.height, self.timestamp_ms
        )
    }
}

/// Python wrapper for a batch of frames.
///
/// Provides efficient batched access to frame data for training.
///
/// # Example
///
/// ```python
/// from drivedit import FrameBatch
///
/// batch.batch_size    # int
/// batch.width         # int
/// batch.height        # int
/// batch.frames        # numpy array [B, H, W, 3] uint8
/// batch.telemetry     # numpy array [B, 20] float32
/// batch.frame_ids     # numpy array [B] uint64
/// batch.timestamps    # numpy array [B] float64
/// ```
#[pyclass(name = "FrameBatch")]
#[derive(Debug)]
pub struct PyFrameBatch {
    /// Contiguous pixel buffer.
    pixels: Vec<u8>,

    /// Telemetry for each frame.
    telemetry: Vec<f32>,

    /// Frame IDs.
    frame_ids: Vec<u64>,

    /// Timestamps.
    timestamps: Vec<f64>,

    /// Number of frames in batch.
    #[pyo3(get)]
    pub batch_size: usize,

    /// Frame width.
    #[pyo3(get)]
    pub width: u32,

    /// Frame height.
    #[pyo3(get)]
    pub height: u32,
}

#[pymethods]
impl PyFrameBatch {
    /// Create an empty batch with capacity.
    #[new]
    #[pyo3(signature = (capacity = 8, width = 640, height = 360))]
    fn new(capacity: usize, width: u32, height: u32) -> Self {
        let frame_size = (width as usize) * (height as usize) * 3;
        Self {
            pixels: Vec::with_capacity(capacity * frame_size),
            telemetry: Vec::with_capacity(capacity * TELEMETRY_ARRAY_SIZE),
            frame_ids: Vec::with_capacity(capacity),
            timestamps: Vec::with_capacity(capacity),
            batch_size: 0,
            width,
            height,
        }
    }

    /// Get frames as numpy array [B, H, W, 3] uint8.
    fn frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray4<u8>>> {
        let shape = [
            self.batch_size,
            self.height as usize,
            self.width as usize,
            3,
        ];
        let arr = self.pixels.clone().into_pyarray(py);
        arr.reshape(shape)
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Get telemetry as numpy array [B, 20] float32.
    fn telemetry<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = [self.batch_size, TELEMETRY_ARRAY_SIZE];
        let arr = self.telemetry.clone().into_pyarray(py);
        arr.reshape(shape)
            .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
    }

    /// Get frame IDs as numpy array [B] uint64.
    fn frame_ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        self.frame_ids.clone().into_pyarray(py)
    }

    /// Get timestamps as numpy array [B] float64.
    fn timestamps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.timestamps.clone().into_pyarray(py)
    }

    /// Convert to dictionary for PyTorch DataLoader compatibility.
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        dict.set_item("frames", self.frames(py)?)?;
        dict.set_item("telemetry", self.telemetry(py)?)?;
        dict.set_item("frame_ids", self.frame_ids(py))?;
        dict.set_item("timestamps", self.timestamps(py))?;
        dict.set_item("batch_size", self.batch_size)?;
        dict.set_item("width", self.width)?;
        dict.set_item("height", self.height)?;
        Ok(dict)
    }

    /// Check if batch is empty.
    fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    fn __len__(&self) -> usize {
        self.batch_size
    }

    fn __repr__(&self) -> String {
        format!(
            "FrameBatch(size={}, {}x{})",
            self.batch_size, self.width, self.height
        )
    }
}

impl PyFrameBatch {
    /// Add a frame to the batch (internal use).
    pub fn push(&mut self, frame: PyFrame) {
        self.pixels.extend_from_slice(&frame.pixels);

        // Convert telemetry to array and extend
        let tel_arr = [
            frame.telemetry.pos_x,
            frame.telemetry.pos_y,
            frame.telemetry.pos_z,
            frame.telemetry.forward_x,
            frame.telemetry.forward_y,
            frame.telemetry.forward_z,
            frame.telemetry.up_x,
            frame.telemetry.up_y,
            frame.telemetry.up_z,
            frame.telemetry.right_x,
            frame.telemetry.right_y,
            frame.telemetry.right_z,
            frame.telemetry.speed_kmh,
            frame.telemetry.steering,
            frame.telemetry.throttle,
            frame.telemetry.brake,
            frame.telemetry.acceleration,
            frame.telemetry.distance_total,
            frame.telemetry.heading_rad,
            frame.telemetry.heading_rate,
        ];
        self.telemetry.extend_from_slice(&tel_arr);
        self.frame_ids.push(frame.frame_id);
        self.timestamps.push(frame.timestamp_ms);
        self.batch_size += 1;
    }

    /// Clear the batch for reuse.
    pub fn clear(&mut self) {
        self.pixels.clear();
        self.telemetry.clear();
        self.frame_ids.clear();
        self.timestamps.clear();
        self.batch_size = 0;
    }
}

/// Register types in the types submodule.
pub fn register_types_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTelemetry>()?;
    m.add_class::<PyFrame>()?;
    m.add_class::<PyFrameBatch>()?;
    m.add("TELEMETRY_SIZE", TELEMETRY_ARRAY_SIZE)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_default() {
        let tel = PyTelemetry::default();
        assert_eq!(tel.speed_kmh, 0.0);
        assert_eq!(tel.steering, 0.0);
    }

    #[test]
    fn test_telemetry_from_array() {
        let arr = vec![0.0f32; TELEMETRY_ARRAY_SIZE];
        let tel = PyTelemetry::from_array(arr).unwrap();
        assert_eq!(tel.speed_kmh, 0.0);
    }

    #[test]
    fn test_frame_batch() {
        let mut batch = PyFrameBatch::new(8, 640, 360);
        assert!(batch.is_empty());
        assert_eq!(batch.batch_size, 0);
    }
}
