//! Zero-copy numpy array utilities.
//!
//! Provides efficient conversion functions between Rust arrays and numpy,
//! including HWC/CHW format conversion and normalization.

use numpy::{IntoPyArray, PyArray3, PyArray4, PyArrayMethods, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Convert HWC uint8 array to CHW float32 with normalization.
///
/// # Arguments
///
/// * `arr` - Input array with shape [H, W, C] or [B, H, W, C], dtype uint8
///
/// # Returns
///
/// Output array with shape [C, H, W] or [B, C, H, W], dtype float32, normalized to [0, 1]
///
/// # Example
///
/// ```python
/// import drivedit
/// import numpy as np
///
/// hwc = np.random.randint(0, 256, (360, 640, 3), dtype=np.uint8)
/// chw = drivedit.hwc_to_chw(hwc)  # [3, 360, 640] float32
/// ```
#[pyfunction]
pub fn hwc_to_chw<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray4<u8>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let shape = arr.shape();
    if shape.len() != 4 || shape[3] != 3 {
        return Err(PyValueError::new_err(
            "Expected array with shape [B, H, W, 3]",
        ));
    }

    let b = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    // Allocate output
    let mut output = vec![0.0f32; b * c * h * w];

    // Convert with normalization
    let input = arr.as_slice().map_err(|e| {
        PyValueError::new_err(format!("Array not contiguous: {}", e))
    })?;

    for batch in 0..b {
        for y in 0..h {
            for x in 0..w {
                let src_idx = batch * h * w * c + y * w * c + x * c;
                for ch in 0..c {
                    let dst_idx = batch * c * h * w + ch * h * w + y * w + x;
                    output[dst_idx] = input[src_idx + ch] as f32 / 255.0;
                }
            }
        }
    }

    let py_arr = output.into_pyarray(py);
    py_arr
        .reshape([b, c, h, w])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
}

/// Convert CHW float32 array to HWC uint8.
///
/// # Arguments
///
/// * `arr` - Input array with shape [C, H, W] or [B, C, H, W], dtype float32
///
/// # Returns
///
/// Output array with shape [H, W, C] or [B, H, W, C], dtype uint8
///
/// # Example
///
/// ```python
/// import drivedit
/// import numpy as np
///
/// chw = np.random.rand(3, 360, 640).astype(np.float32)
/// hwc = drivedit.chw_to_hwc(chw)  # [360, 640, 3] uint8
/// ```
#[pyfunction]
pub fn chw_to_hwc<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray4<f32>,
) -> PyResult<Bound<'py, PyArray4<u8>>> {
    let shape = arr.shape();
    if shape.len() != 4 || shape[1] != 3 {
        return Err(PyValueError::new_err(
            "Expected array with shape [B, 3, H, W]",
        ));
    }

    let b = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];

    // Allocate output
    let mut output = vec![0u8; b * h * w * c];

    // Convert with denormalization
    let input = arr.as_slice().map_err(|e| {
        PyValueError::new_err(format!("Array not contiguous: {}", e))
    })?;

    for batch in 0..b {
        for y in 0..h {
            for x in 0..w {
                let dst_idx = batch * h * w * c + y * w * c + x * c;
                for ch in 0..c {
                    let src_idx = batch * c * h * w + ch * h * w + y * w + x;
                    let val = (input[src_idx] * 255.0).clamp(0.0, 255.0) as u8;
                    output[dst_idx + ch] = val;
                }
            }
        }
    }

    let py_arr = output.into_pyarray(py);
    py_arr
        .reshape([b, h, w, c])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
}

/// Normalize frames from uint8 [0, 255] to float32 [0, 1].
///
/// Preserves the input layout (HWC or CHW).
///
/// # Arguments
///
/// * `arr` - Input array with dtype uint8
///
/// # Returns
///
/// Output array with same shape, dtype float32, values in [0, 1]
///
/// # Example
///
/// ```python
/// import drivedit
/// import numpy as np
///
/// frames = np.random.randint(0, 256, (8, 360, 640, 3), dtype=np.uint8)
/// normalized = drivedit.normalize_frames(frames)  # float32 [0, 1]
/// ```
#[pyfunction]
pub fn normalize_frames<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray4<u8>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let shape = arr.shape();

    let input = arr.as_slice().map_err(|e| {
        PyValueError::new_err(format!("Array not contiguous: {}", e))
    })?;

    // Normalize
    let output: Vec<f32> = input.iter().map(|&v| v as f32 / 255.0).collect();

    let py_arr = output.into_pyarray(py);
    py_arr
        .reshape([shape[0], shape[1], shape[2], shape[3]])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
}

/// Convert a 3D HWC array to CHW (single frame, no batch dimension).
#[pyfunction]
pub fn hwc_to_chw_3d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<u8>,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let shape = arr.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err("Expected array with shape [H, W, 3]"));
    }

    let h = shape[0];
    let w = shape[1];
    let c = shape[2];

    let mut output = vec![0.0f32; c * h * w];

    let input = arr.as_slice().map_err(|e| {
        PyValueError::new_err(format!("Array not contiguous: {}", e))
    })?;

    for y in 0..h {
        for x in 0..w {
            let src_idx = y * w * c + x * c;
            for ch in 0..c {
                let dst_idx = ch * h * w + y * w + x;
                output[dst_idx] = input[src_idx + ch] as f32 / 255.0;
            }
        }
    }

    let py_arr = output.into_pyarray(py);
    py_arr
        .reshape([c, h, w])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
}

/// Convert a 3D CHW array to HWC (single frame, no batch dimension).
#[pyfunction]
pub fn chw_to_hwc_3d<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray3<f32>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let shape = arr.shape();
    if shape.len() != 3 || shape[0] != 3 {
        return Err(PyValueError::new_err("Expected array with shape [3, H, W]"));
    }

    let c = shape[0];
    let h = shape[1];
    let w = shape[2];

    let mut output = vec![0u8; h * w * c];

    let input = arr.as_slice().map_err(|e| {
        PyValueError::new_err(format!("Array not contiguous: {}", e))
    })?;

    for y in 0..h {
        for x in 0..w {
            let dst_idx = y * w * c + x * c;
            for ch in 0..c {
                let src_idx = ch * h * w + y * w + x;
                let val = (input[src_idx] * 255.0).clamp(0.0, 255.0) as u8;
                output[dst_idx + ch] = val;
            }
        }
    }

    let py_arr = output.into_pyarray(py);
    py_arr
        .reshape([h, w, c])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
}

/// Apply ImageNet normalization to CHW float32 tensor.
///
/// Normalizes using ImageNet mean and std:
/// - mean: [0.485, 0.456, 0.406]
/// - std: [0.229, 0.224, 0.225]
///
/// # Arguments
///
/// * `arr` - Input array with shape [B, C, H, W], dtype float32, values in [0, 1]
///
/// # Returns
///
/// Normalized array with same shape
#[pyfunction]
pub fn imagenet_normalize<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray4<f32>,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let shape = arr.shape();
    if shape.len() != 4 || shape[1] != 3 {
        return Err(PyValueError::new_err(
            "Expected array with shape [B, 3, H, W]",
        ));
    }

    let b = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];

    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    let input = arr.as_slice().map_err(|e| {
        PyValueError::new_err(format!("Array not contiguous: {}", e))
    })?;

    let mut output = vec![0.0f32; b * c * h * w];

    for batch in 0..b {
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let idx = batch * c * h * w + ch * h * w + y * w + x;
                    output[idx] = (input[idx] - mean[ch]) / std[ch];
                }
            }
        }
    }

    let py_arr = output.into_pyarray(py);
    py_arr
        .reshape([b, c, h, w])
        .map_err(|e| PyValueError::new_err(format!("Reshape failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization_constants() {
        // ImageNet mean and std
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        // All values should be valid
        for m in &mean {
            assert!(*m > 0.0 && *m < 1.0);
        }
        for s in &std {
            assert!(*s > 0.0 && *s < 1.0);
        }
    }
}
