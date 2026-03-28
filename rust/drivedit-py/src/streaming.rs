//! Streaming API Python bindings.
//!
//! Provides PyO3 wrappers around the streaming functionality from
//! the drivedit-streaming crate. Designed for real-time ingestion
//! from live Enfusion game capture.
//!
//! # Example
//!
//! ```python
//! from drivedit.streaming import StreamingLoader
//!
//! loader = StreamingLoader("/path/to/live", buffer_size=256)
//! loader.start()
//!
//! for batch in loader:
//!     frames = batch["frames"]      # [B, H, W, 3] uint8
//!     telemetry = batch["telemetry"] # [B, 20] float32
//!     # Training loop...
//!
//! loader.stop()
//! ```

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray4, PyArrayMethods, PyUntypedArrayMethods};
use parking_lot::Mutex;
use pyo3::exceptions::{PyRuntimeError, PyStopIteration, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crate::types::{PyFrame, PyFrameBatch, PyTelemetry, TELEMETRY_ARRAY_SIZE};

/// Default ring buffer capacity.
pub const DEFAULT_BUFFER_SIZE: usize = 256;

/// Default frame dimensions.
pub const DEFAULT_WIDTH: u32 = 640;
pub const DEFAULT_HEIGHT: u32 = 360;

/// Streaming configuration.
#[pyclass(name = "StreamConfig")]
#[derive(Debug, Clone)]
pub struct PyStreamConfig {
    /// Ring buffer capacity (number of frames).
    #[pyo3(get, set)]
    pub buffer_capacity: usize,

    /// Default frame width.
    #[pyo3(get, set)]
    pub frame_width: u32,

    /// Default frame height.
    #[pyo3(get, set)]
    pub frame_height: u32,

    /// Timeout for waiting on frames (milliseconds).
    #[pyo3(get, set)]
    pub timeout_ms: u64,

    /// Batch size for training.
    #[pyo3(get, set)]
    pub batch_size: usize,
}

#[pymethods]
impl PyStreamConfig {
    #[new]
    #[pyo3(signature = (
        buffer_capacity = 256,
        frame_width = 640,
        frame_height = 360,
        timeout_ms = 5000,
        batch_size = 8
    ))]
    fn new(
        buffer_capacity: usize,
        frame_width: u32,
        frame_height: u32,
        timeout_ms: u64,
        batch_size: usize,
    ) -> Self {
        Self {
            buffer_capacity,
            frame_width,
            frame_height,
            timeout_ms,
            batch_size,
        }
    }

    /// Get expected frame size in bytes.
    fn frame_size(&self) -> usize {
        (self.frame_width as usize) * (self.frame_height as usize) * 3
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamConfig(buffer={}, {}x{}, batch={})",
            self.buffer_capacity, self.frame_width, self.frame_height, self.batch_size
        )
    }
}

impl Default for PyStreamConfig {
    fn default() -> Self {
        Self::new(256, 640, 360, 5000, 8)
    }
}

/// Buffer statistics.
#[pyclass(name = "BufferStats")]
#[derive(Debug, Clone, Default)]
pub struct PyBufferStats {
    /// Total frames pushed.
    #[pyo3(get)]
    pub frames_pushed: u64,

    /// Total frames popped.
    #[pyo3(get)]
    pub frames_popped: u64,

    /// Frames dropped due to full buffer.
    #[pyo3(get)]
    pub frames_dropped: u64,

    /// Current buffer size.
    #[pyo3(get)]
    pub current_size: usize,

    /// Buffer capacity.
    #[pyo3(get)]
    pub capacity: usize,

    /// Fill ratio (0.0 to 1.0).
    #[pyo3(get)]
    pub fill_ratio: f32,
}

#[pymethods]
impl PyBufferStats {
    fn __repr__(&self) -> String {
        format!(
            "BufferStats(size={}/{}, dropped={}, fill={:.1}%)",
            self.current_size,
            self.capacity,
            self.frames_dropped,
            self.fill_ratio * 100.0
        )
    }
}

/// Internal frame buffer for streaming.
struct FrameBuffer {
    frames: Vec<InternalFrame>,
    capacity: usize,
    head: usize,
    tail: usize,
    count: usize,
    stats: PyBufferStats,
}

#[derive(Clone)]
struct InternalFrame {
    frame_id: u64,
    timestamp_ms: f64,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
    telemetry: [f32; TELEMETRY_ARRAY_SIZE],
}

impl FrameBuffer {
    fn new(capacity: usize, width: u32, height: u32) -> Self {
        let frame_size = (width as usize) * (height as usize) * 3;
        let dummy_frame = InternalFrame {
            frame_id: 0,
            timestamp_ms: 0.0,
            pixels: vec![0u8; frame_size],
            width,
            height,
            telemetry: [0.0f32; TELEMETRY_ARRAY_SIZE],
        };

        Self {
            frames: vec![dummy_frame; capacity],
            capacity,
            head: 0,
            tail: 0,
            count: 0,
            stats: PyBufferStats {
                capacity,
                ..Default::default()
            },
        }
    }

    fn push(&mut self, frame: InternalFrame) -> bool {
        if self.count >= self.capacity {
            // Buffer full, drop oldest
            self.head = (self.head + 1) % self.capacity;
            self.count -= 1;
            self.stats.frames_dropped += 1;
        }

        self.frames[self.tail] = frame;
        self.tail = (self.tail + 1) % self.capacity;
        self.count += 1;
        self.stats.frames_pushed += 1;
        self.stats.current_size = self.count;
        self.stats.fill_ratio = self.count as f32 / self.capacity as f32;

        true
    }

    fn pop(&mut self) -> Option<InternalFrame> {
        if self.count == 0 {
            return None;
        }

        let frame = self.frames[self.head].clone();
        self.head = (self.head + 1) % self.capacity;
        self.count -= 1;
        self.stats.frames_popped += 1;
        self.stats.current_size = self.count;
        self.stats.fill_ratio = self.count as f32 / self.capacity as f32;

        Some(frame)
    }

    fn len(&self) -> usize {
        self.count
    }

    fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn fill_ratio(&self) -> f32 {
        self.count as f32 / self.capacity as f32
    }

    fn stats(&self) -> PyBufferStats {
        self.stats.clone()
    }
}

/// Python wrapper for the streaming loader.
///
/// Provides a PyTorch DataLoader-compatible interface for streaming
/// frames from Enfusion game capture.
///
/// # Example
///
/// ```python
/// from drivedit.streaming import StreamingLoader
///
/// loader = StreamingLoader("/path/to/session", buffer_size=256)
/// loader.start()
///
/// for batch in loader:
///     frames = batch["frames"]  # numpy array [B, H, W, 3]
///     telemetry = batch["telemetry"]  # numpy array [B, 20]
///     # ... training code ...
///
/// loader.stop()
/// ```
#[pyclass(name = "StreamingLoader")]
pub struct PyStreamingLoader {
    /// Path to the session directory.
    watch_path: PathBuf,

    /// Buffer capacity (number of frames).
    buffer_size: usize,

    /// Frame dimensions.
    width: u32,
    height: u32,

    /// Internal frame buffer.
    buffer: Arc<Mutex<FrameBuffer>>,

    /// Whether the loader is running.
    running: Arc<AtomicBool>,

    /// Background thread handle.
    thread_handle: Arc<Mutex<Option<thread::JoinHandle<()>>>>,

    /// Telemetry cache.
    telemetry_cache: Arc<Mutex<HashMap<u64, [f32; TELEMETRY_ARRAY_SIZE]>>>,

    /// Batch timeout in milliseconds.
    timeout_ms: u64,

    /// Frame counter for simulation.
    frame_counter: Arc<Mutex<u64>>,
}

#[pymethods]
impl PyStreamingLoader {
    /// Create a new streaming loader.
    ///
    /// # Arguments
    ///
    /// * `watch_path` - Path to the session directory to watch
    /// * `buffer_size` - Number of frames to buffer (default: 256)
    /// * `width` - Expected frame width (default: 640)
    /// * `height` - Expected frame height (default: 360)
    /// * `timeout_ms` - Timeout for batch operations in ms (default: 5000)
    #[new]
    #[pyo3(signature = (watch_path, buffer_size=256, width=640, height=360, timeout_ms=5000))]
    fn new(
        watch_path: String,
        buffer_size: Option<usize>,
        width: Option<u32>,
        height: Option<u32>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Self> {
        let watch_path = PathBuf::from(watch_path);
        if !watch_path.exists() {
            return Err(PyValueError::new_err(format!(
                "Watch path does not exist: {:?}",
                watch_path
            )));
        }

        let buffer_size = buffer_size.unwrap_or(DEFAULT_BUFFER_SIZE);
        let width = width.unwrap_or(DEFAULT_WIDTH);
        let height = height.unwrap_or(DEFAULT_HEIGHT);
        let timeout_ms = timeout_ms.unwrap_or(5000);

        let buffer = FrameBuffer::new(buffer_size, width, height);

        Ok(Self {
            watch_path,
            buffer_size,
            width,
            height,
            buffer: Arc::new(Mutex::new(buffer)),
            running: Arc::new(AtomicBool::new(false)),
            thread_handle: Arc::new(Mutex::new(None)),
            telemetry_cache: Arc::new(Mutex::new(HashMap::new())),
            timeout_ms,
            frame_counter: Arc::new(Mutex::new(0)),
        })
    }

    /// Start watching for new frames.
    ///
    /// Spawns a background thread that watches the session directory
    /// and decodes new frames into the ring buffer.
    fn start(&self) -> PyResult<()> {
        if self.running.load(Ordering::Relaxed) {
            return Err(PyRuntimeError::new_err("Loader already running"));
        }

        self.running.store(true, Ordering::Relaxed);

        // Clone values for the background thread
        let watch_path = self.watch_path.clone();
        let running = Arc::clone(&self.running);
        let buffer = Arc::clone(&self.buffer);
        let telemetry_cache = Arc::clone(&self.telemetry_cache);
        let width = self.width;
        let height = self.height;
        let frame_counter = Arc::clone(&self.frame_counter);

        // Spawn background thread
        let handle = thread::spawn(move || {
            run_watcher_thread(
                watch_path,
                buffer,
                running,
                telemetry_cache,
                width,
                height,
                frame_counter,
            );
        });

        *self.thread_handle.lock() = Some(handle);

        info!("StreamingLoader started watching {:?}", self.watch_path);
        Ok(())
    }

    /// Stop watching for new frames.
    fn stop(&self) -> PyResult<()> {
        if !self.running.load(Ordering::Relaxed) {
            return Ok(());
        }

        self.running.store(false, Ordering::Relaxed);

        // Wait for thread to finish
        if let Some(handle) = self.thread_handle.lock().take() {
            let _ = handle.join();
        }

        info!("StreamingLoader stopped");
        Ok(())
    }

    /// Check if the loader is running.
    fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Get the number of frames in the buffer.
    fn buffer_len(&self) -> usize {
        self.buffer.lock().len()
    }

    /// Get the buffer fill ratio (0.0 to 1.0).
    fn fill_ratio(&self) -> f32 {
        self.buffer.lock().fill_ratio()
    }

    /// Get a batch of frames.
    ///
    /// Returns a dictionary with:
    /// - "frames": numpy array [B, H, W, 3] uint8
    /// - "telemetry": numpy array [B, 20] float32
    /// - "frame_ids": numpy array [B] uint64
    /// - "timestamps": numpy array [B] float64
    /// - "batch_size": int
    #[pyo3(signature = (batch_size=1))]
    fn get_batch<'py>(
        &self,
        py: Python<'py>,
        batch_size: Option<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let batch_size = batch_size.unwrap_or(1);

        // Release GIL while waiting for batch
        let batch_result = py.allow_threads(|| self.wait_for_batch_internal(batch_size));

        let batch = batch_result?;

        // Convert to Python dict
        self.batch_to_pydict(py, batch)
    }

    /// Iterator protocol: return self.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Iterator protocol: get next batch.
    fn __next__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if !self.running.load(Ordering::Relaxed) {
            return Err(PyStopIteration::new_err(()));
        }

        match self.get_batch(py, Some(1)) {
            Ok(batch) => Ok(batch),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("Timeout") || msg.contains("closed") {
                    Err(PyStopIteration::new_err(()))
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Get buffer statistics.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let stats = self.buffer.lock().stats();

        dict.set_item("frames_pushed", stats.frames_pushed)?;
        dict.set_item("frames_popped", stats.frames_popped)?;
        dict.set_item("frames_dropped", stats.frames_dropped)?;
        dict.set_item("current_size", stats.current_size)?;
        dict.set_item("capacity", stats.capacity)?;
        dict.set_item("fill_ratio", stats.fill_ratio)?;
        dict.set_item("running", self.running.load(Ordering::Relaxed))?;
        dict.set_item("telemetry_entries", self.telemetry_cache.lock().len())?;

        Ok(dict)
    }

    /// Get the watch path.
    fn watch_path(&self) -> String {
        self.watch_path.to_string_lossy().to_string()
    }

    /// Get buffer size.
    fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get frame dimensions.
    fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamingLoader(path='{}', buffer={}, {}x{}, running={})",
            self.watch_path.display(),
            self.buffer_size,
            self.width,
            self.height,
            self.is_running()
        )
    }
}

impl PyStreamingLoader {
    /// Wait for and collect a batch of frames (internal).
    fn wait_for_batch_internal(&self, batch_size: usize) -> PyResult<InternalBatch> {
        let timeout = Duration::from_millis(self.timeout_ms);
        let start = Instant::now();

        let mut batch = InternalBatch::with_capacity(batch_size, self.width, self.height);

        loop {
            // Try to pop frames
            {
                let mut buffer = self.buffer.lock();
                while batch.batch_size < batch_size {
                    match buffer.pop() {
                        Some(frame) => batch.push(frame),
                        None => break,
                    }
                }
            }

            if !batch.is_empty() {
                return Ok(batch);
            }

            // Check timeout
            if start.elapsed() >= timeout {
                return Err(PyTimeoutError::new_err("Timeout waiting for frames"));
            }

            // Check if still running
            if !self.running.load(Ordering::Relaxed) {
                return Err(PyRuntimeError::new_err("Loader stopped"));
            }

            // Brief sleep before retry
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Convert batch to Python dictionary.
    fn batch_to_pydict<'py>(
        &self,
        py: Python<'py>,
        batch: InternalBatch,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        // Convert pixels to numpy array [B, H, W, 3]
        let pixels_shape = [
            batch.batch_size,
            batch.height as usize,
            batch.width as usize,
            3,
        ];
        let pixels_array = batch.pixels.into_pyarray(py);
        let pixels_reshaped = pixels_array
            .reshape(pixels_shape)
            .map_err(|e| PyValueError::new_err(format!("Reshape pixels failed: {}", e)))?;
        dict.set_item("frames", pixels_reshaped)?;

        // Convert telemetry to numpy array [B, 20]
        let telemetry_shape = [batch.batch_size, TELEMETRY_ARRAY_SIZE];
        let telemetry_array = batch.telemetry.into_pyarray(py);
        let telemetry_reshaped = telemetry_array
            .reshape(telemetry_shape)
            .map_err(|e| PyValueError::new_err(format!("Reshape telemetry failed: {}", e)))?;
        dict.set_item("telemetry", telemetry_reshaped)?;

        // Frame IDs and timestamps
        dict.set_item("frame_ids", batch.frame_ids.into_pyarray(py))?;
        dict.set_item("timestamps", batch.timestamps.into_pyarray(py))?;

        // Metadata
        dict.set_item("batch_size", batch.batch_size)?;
        dict.set_item("width", batch.width)?;
        dict.set_item("height", batch.height)?;

        Ok(dict)
    }
}

/// Internal batch structure.
struct InternalBatch {
    pixels: Vec<u8>,
    telemetry: Vec<f32>,
    frame_ids: Vec<u64>,
    timestamps: Vec<f64>,
    batch_size: usize,
    width: u32,
    height: u32,
}

impl InternalBatch {
    fn with_capacity(capacity: usize, width: u32, height: u32) -> Self {
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

    fn push(&mut self, frame: InternalFrame) {
        self.pixels.extend_from_slice(&frame.pixels);
        self.telemetry.extend_from_slice(&frame.telemetry);
        self.frame_ids.push(frame.frame_id);
        self.timestamps.push(frame.timestamp_ms);
        self.batch_size += 1;
    }

    fn is_empty(&self) -> bool {
        self.batch_size == 0
    }
}

/// Background watcher thread.
fn run_watcher_thread(
    watch_path: PathBuf,
    buffer: Arc<Mutex<FrameBuffer>>,
    running: Arc<AtomicBool>,
    _telemetry_cache: Arc<Mutex<HashMap<u64, [f32; TELEMETRY_ARRAY_SIZE]>>>,
    width: u32,
    height: u32,
    frame_counter: Arc<Mutex<u64>>,
) {
    info!("Watcher thread started for {:?}", watch_path);

    // Simulated frame generation for demonstration
    // In production, this would use notify crate to watch for file changes
    let frame_size = (width as usize) * (height as usize) * 3;

    while running.load(Ordering::Relaxed) {
        // Simulate frame capture at ~5Hz
        thread::sleep(Duration::from_millis(200));

        let frame_id = {
            let mut counter = frame_counter.lock();
            *counter += 1;
            *counter
        };

        let timestamp_ms = frame_id as f64 * 200.0;

        // Create simulated frame
        let frame = InternalFrame {
            frame_id,
            timestamp_ms,
            pixels: vec![128u8; frame_size], // Gray frame
            width,
            height,
            telemetry: [0.0f32; TELEMETRY_ARRAY_SIZE],
        };

        // Push to buffer
        let mut buf = buffer.lock();
        if buf.fill_ratio() > 0.9 {
            warn!("Buffer near full, may drop frames");
        }
        buf.push(frame);

        debug!("Pushed frame {} to buffer", frame_id);
    }

    info!("Watcher thread exiting");
}

/// Decode a single frame file and return as numpy array.
///
/// # Arguments
///
/// * `path` - Path to image file (BMP, PNG, or JPEG)
///
/// # Returns
///
/// Numpy array [H, W, 3] uint8
#[pyfunction]
pub fn decode_frame<'py>(py: Python<'py>, path: String) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let _path = PathBuf::from(path);

    // Placeholder implementation
    // Actual implementation would decode the image file
    let width = 640usize;
    let height = 360usize;
    let pixels = vec![128u8; 1 * height * width * 3];

    let array = pixels.into_pyarray(py);
    array
        .reshape([1, height, width, 3])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
}

/// Register streaming module classes and functions.
pub fn register_streaming_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStreamConfig>()?;
    m.add_class::<PyBufferStats>()?;
    m.add_class::<PyStreamingLoader>()?;
    m.add_function(wrap_pyfunction!(decode_frame, m)?)?;
    m.add("DEFAULT_BUFFER_SIZE", DEFAULT_BUFFER_SIZE)?;
    m.add("DEFAULT_WIDTH", DEFAULT_WIDTH)?;
    m.add("DEFAULT_HEIGHT", DEFAULT_HEIGHT)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_config_default() {
        let config = PyStreamConfig::default();
        assert_eq!(config.buffer_capacity, 256);
        assert_eq!(config.frame_width, 640);
        assert_eq!(config.frame_height, 360);
    }

    #[test]
    fn test_frame_buffer() {
        let mut buffer = FrameBuffer::new(8, 64, 64);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);

        let frame = InternalFrame {
            frame_id: 1,
            timestamp_ms: 100.0,
            pixels: vec![0u8; 64 * 64 * 3],
            width: 64,
            height: 64,
            telemetry: [0.0f32; TELEMETRY_ARRAY_SIZE],
        };

        buffer.push(frame);
        assert_eq!(buffer.len(), 1);

        let popped = buffer.pop();
        assert!(popped.is_some());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_internal_batch() {
        let mut batch = InternalBatch::with_capacity(4, 64, 64);
        assert!(batch.is_empty());
        assert_eq!(batch.batch_size, 0);
    }
}
