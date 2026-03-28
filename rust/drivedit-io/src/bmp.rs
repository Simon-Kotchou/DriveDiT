//! Zero-dependency BMP decoder for real-time frame loading.
//!
//! This decoder is optimized for the specific BMP format output by Enfusion
//! engine capture: 24-bit BGR, bottom-up, uncompressed.
//!
//! # Performance
//!
//! - Zero allocations for header parsing
//! - Single-pass pixel conversion (BGR -> RGB, flip)
//! - No external dependencies
//!
//! # Supported Formats
//!
//! - 24-bit BGR (BI_RGB)
//! - 32-bit BGRA (BI_RGB)
//! - Bottom-up or top-down orientation

use crate::error::{IoError, Result};
use drivedit_types::{RawFrame, Telemetry};
use image::{GenericImageView, ImageReader};
use std::path::Path;

/// BMP file header (14 bytes).
#[derive(Debug, Clone, Copy)]
struct BmpFileHeader {
    /// Signature - should be "BM".
    #[allow(dead_code)]
    signature: [u8; 2],
    /// File size in bytes.
    #[allow(dead_code)]
    file_size: u32,
    /// Reserved (unused).
    #[allow(dead_code)]
    reserved: u32,
    /// Offset to pixel data.
    pixel_offset: u32,
}

/// BMP info header (BITMAPINFOHEADER, 40 bytes).
#[derive(Debug, Clone, Copy)]
struct BmpInfoHeader {
    /// Header size (should be 40 for BITMAPINFOHEADER).
    #[allow(dead_code)]
    header_size: u32,
    /// Image width in pixels.
    width: i32,
    /// Image height in pixels (negative = top-down).
    height: i32,
    /// Number of color planes (should be 1).
    #[allow(dead_code)]
    planes: u16,
    /// Bits per pixel (24 or 32).
    bits_per_pixel: u16,
    /// Compression method (0 = BI_RGB, uncompressed).
    compression: u32,
    /// Image size (may be 0 for BI_RGB).
    #[allow(dead_code)]
    image_size: u32,
    /// Horizontal resolution (pixels/meter).
    #[allow(dead_code)]
    x_pixels_per_meter: i32,
    /// Vertical resolution (pixels/meter).
    #[allow(dead_code)]
    y_pixels_per_meter: i32,
    /// Number of colors in palette.
    #[allow(dead_code)]
    colors_used: u32,
    /// Number of important colors.
    #[allow(dead_code)]
    colors_important: u32,
}

const BMP_FILE_HEADER_SIZE: usize = 14;
const BMP_INFO_HEADER_SIZE: usize = 40;
const BI_RGB: u32 = 0;

/// Parse the BMP file header.
fn parse_file_header(data: &[u8]) -> Result<BmpFileHeader> {
    if data.len() < BMP_FILE_HEADER_SIZE {
        return Err(IoError::BufferTooSmall {
            expected: BMP_FILE_HEADER_SIZE,
            actual: data.len(),
        });
    }

    // Check signature
    if data[0] != b'B' || data[1] != b'M' {
        return Err(IoError::InvalidBmpHeader(format!(
            "Invalid signature: {:02X}{:02X}",
            data[0], data[1]
        )));
    }

    Ok(BmpFileHeader {
        signature: [data[0], data[1]],
        file_size: u32::from_le_bytes([data[2], data[3], data[4], data[5]]),
        reserved: u32::from_le_bytes([data[6], data[7], data[8], data[9]]),
        pixel_offset: u32::from_le_bytes([data[10], data[11], data[12], data[13]]),
    })
}

/// Parse the BMP info header.
fn parse_info_header(data: &[u8]) -> Result<BmpInfoHeader> {
    if data.len() < BMP_INFO_HEADER_SIZE {
        return Err(IoError::BufferTooSmall {
            expected: BMP_INFO_HEADER_SIZE,
            actual: data.len(),
        });
    }

    let header_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);

    // We only support BITMAPINFOHEADER (40 bytes) for now
    if header_size < 40 {
        return Err(IoError::UnsupportedBmpFormat(format!(
            "Unsupported header size: {}",
            header_size
        )));
    }

    Ok(BmpInfoHeader {
        header_size,
        width: i32::from_le_bytes([data[4], data[5], data[6], data[7]]),
        height: i32::from_le_bytes([data[8], data[9], data[10], data[11]]),
        planes: u16::from_le_bytes([data[12], data[13]]),
        bits_per_pixel: u16::from_le_bytes([data[14], data[15]]),
        compression: u32::from_le_bytes([data[16], data[17], data[18], data[19]]),
        image_size: u32::from_le_bytes([data[20], data[21], data[22], data[23]]),
        x_pixels_per_meter: i32::from_le_bytes([data[24], data[25], data[26], data[27]]),
        y_pixels_per_meter: i32::from_le_bytes([data[28], data[29], data[30], data[31]]),
        colors_used: u32::from_le_bytes([data[32], data[33], data[34], data[35]]),
        colors_important: u32::from_le_bytes([data[36], data[37], data[38], data[39]]),
    })
}

/// Decode a BMP image from raw bytes.
///
/// Returns a [`RawFrame`] with RGB pixel data in row-major [H, W, 3] layout.
///
/// # Supported Formats
///
/// - 24-bit BGR uncompressed
/// - 32-bit BGRA uncompressed
/// - Bottom-up or top-down orientation
///
/// # Example
///
/// ```rust,ignore
/// use drivedit_io::decode_bmp;
/// use std::fs;
///
/// let bytes = fs::read("frame_00001.bmp")?;
/// let frame = decode_bmp(&bytes)?;
/// assert!(frame.is_valid());
/// ```
pub fn decode_bmp(data: &[u8]) -> Result<RawFrame> {
    // Parse headers
    let file_header = parse_file_header(data)?;
    let info_header = parse_info_header(&data[BMP_FILE_HEADER_SIZE..])?;

    // Validate dimensions
    if info_header.width <= 0 {
        return Err(IoError::InvalidDimensions {
            width: info_header.width,
            height: info_header.height,
        });
    }

    let width = info_header.width as u32;
    let height = info_header.height.unsigned_abs();
    let is_top_down = info_header.height < 0;

    // Check compression - only support uncompressed
    if info_header.compression != BI_RGB {
        return Err(IoError::UnsupportedBmpFormat(format!(
            "Unsupported compression: {}",
            info_header.compression
        )));
    }

    // Check bits per pixel
    let bytes_per_pixel = match info_header.bits_per_pixel {
        24 => 3,
        32 => 4,
        other => {
            return Err(IoError::UnsupportedBmpFormat(format!(
                "Unsupported bits per pixel: {}",
                other
            )));
        }
    };

    // Calculate row stride (rows are padded to 4-byte boundary)
    let row_bytes = (width as usize) * bytes_per_pixel;
    let row_stride = (row_bytes + 3) & !3;

    // Validate data size
    let pixel_offset = file_header.pixel_offset as usize;
    let expected_size = pixel_offset + row_stride * (height as usize);

    if data.len() < expected_size {
        return Err(IoError::BufferTooSmall {
            expected: expected_size,
            actual: data.len(),
        });
    }

    // Decode pixels
    let pixel_data = &data[pixel_offset..];
    let output_size = (width as usize) * (height as usize) * 3;
    let mut pixels = vec![0u8; output_size];

    for y in 0..height as usize {
        // Source row (bottom-up BMPs start from bottom)
        let src_y = if is_top_down {
            y
        } else {
            (height as usize) - 1 - y
        };
        let src_offset = src_y * row_stride;

        // Destination row
        let dst_offset = y * (width as usize) * 3;

        for x in 0..width as usize {
            let src_px = src_offset + x * bytes_per_pixel;
            let dst_px = dst_offset + x * 3;

            // BMP stores as BGR, convert to RGB
            pixels[dst_px] = pixel_data[src_px + 2]; // R
            pixels[dst_px + 1] = pixel_data[src_px + 1]; // G
            pixels[dst_px + 2] = pixel_data[src_px]; // B
        }
    }

    Ok(RawFrame::new(0, 0.0, pixels, width, height, Telemetry::default()))
}

/// Decode a BMP image from raw bytes (alias for decode_bmp).
///
/// This is an alias for `decode_bmp` for API compatibility.
#[inline]
pub fn decode_bmp_bytes(data: &[u8]) -> Result<RawFrame> {
    decode_bmp(data)
}

/// Decode a BMP image from a file path.
///
/// Uses the `image` crate for robust decoding with support for various BMP variants.
pub fn decode_bmp_file(path: &Path) -> Result<RawFrame> {
    let img = ImageReader::open(path)
        .map_err(|e| IoError::FileIo(e.into()))?
        .decode()
        .map_err(|e| IoError::ImageDecode(e.to_string()))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    Ok(RawFrame::new(0, 0.0, pixels, width, height, Telemetry::default()))
}

/// Decode and resize a BMP image.
///
/// Uses nearest-neighbor interpolation for speed.
pub fn decode_and_resize(
    path: &Path,
    target_width: u32,
    target_height: u32,
) -> Result<RawFrame> {
    let img = ImageReader::open(path)
        .map_err(|e| IoError::FileIo(e.into()))?
        .decode()
        .map_err(|e| IoError::ImageDecode(e.to_string()))?;

    let (src_width, src_height) = img.dimensions();

    // Skip resize if already at target size
    if src_width == target_width && src_height == target_height {
        let rgb = img.to_rgb8();
        return Ok(RawFrame::new(0, 0.0, rgb.into_raw(), target_width, target_height, Telemetry::default()));
    }

    // Resize using nearest neighbor (fast)
    let resized = img.resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::Nearest,
    );

    let rgb = resized.to_rgb8();
    Ok(RawFrame::new(0, 0.0, rgb.into_raw(), target_width, target_height, Telemetry::default()))
}

/// Validate that bytes represent a valid BMP.
#[inline]
pub fn is_valid_bmp(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == b'B' && data[1] == b'M'
}

/// Extract BMP dimensions from header without full decode.
///
/// Returns (width, height) or None if invalid.
pub fn bmp_dimensions(data: &[u8]) -> Option<(u32, u32)> {
    if data.len() < 26 || !is_valid_bmp(data) {
        return None;
    }

    // BMP header: width at offset 18, height at offset 22 (little-endian i32)
    let width = i32::from_le_bytes([data[18], data[19], data[20], data[21]]);
    let height = i32::from_le_bytes([data[22], data[23], data[24], data[25]]);

    // Height can be negative (top-down DIB), take absolute value
    Some((width as u32, height.unsigned_abs()))
}

/// Stateful BMP decoder with optional resize support.
///
/// Maintains target dimensions for consistent resize operations,
/// useful when processing a stream of frames that need uniform dimensions.
#[derive(Debug, Clone)]
pub struct BmpDecoder {
    /// Target width for resize (0 = no resize).
    target_width: u32,
    /// Target height for resize (0 = no resize).
    target_height: u32,
    /// Whether to enable resize.
    resize_enabled: bool,
    /// Frame counter for statistics.
    frames_decoded: u64,
}

impl BmpDecoder {
    /// Create a new decoder with target dimensions for resize.
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            target_width,
            target_height,
            resize_enabled: target_width > 0 && target_height > 0,
            frames_decoded: 0,
        }
    }

    /// Create a decoder that passes through original dimensions.
    pub fn passthrough() -> Self {
        Self::new(0, 0)
    }

    /// Create a decoder with standard Enfusion capture dimensions (640x360).
    pub fn enfusion_standard() -> Self {
        Self::new(640, 360)
    }

    /// Decode a BMP file to raw pixels.
    pub fn decode(&mut self, path: &Path) -> Result<RawFrame> {
        self.frames_decoded += 1;

        if self.resize_enabled {
            decode_and_resize(path, self.target_width, self.target_height)
        } else {
            decode_bmp_file(path)
        }
    }

    /// Decode BMP bytes to raw pixels.
    pub fn decode_bytes(&mut self, data: &[u8]) -> Result<RawFrame> {
        self.frames_decoded += 1;
        decode_bmp(data)
    }

    /// Get target dimensions.
    pub fn target_dimensions(&self) -> (u32, u32) {
        (self.target_width, self.target_height)
    }

    /// Check if resize is enabled.
    pub fn is_resize_enabled(&self) -> bool {
        self.resize_enabled
    }

    /// Get number of frames decoded.
    pub fn frames_decoded(&self) -> u64 {
        self.frames_decoded
    }

    /// Set new target dimensions.
    pub fn set_target_dimensions(&mut self, width: u32, height: u32) {
        self.target_width = width;
        self.target_height = height;
        self.resize_enabled = width > 0 && height > 0;
    }
}

impl Default for BmpDecoder {
    fn default() -> Self {
        Self::passthrough()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a minimal valid BMP file.
    fn create_test_bmp(width: u32, height: u32) -> Vec<u8> {
        let row_bytes = (width as usize) * 3;
        let row_stride = (row_bytes + 3) & !3;
        let pixel_data_size = row_stride * (height as usize);
        let file_size = BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE + pixel_data_size;
        let pixel_offset = BMP_FILE_HEADER_SIZE + BMP_INFO_HEADER_SIZE;

        let mut data = vec![0u8; file_size];

        // File header
        data[0] = b'B';
        data[1] = b'M';
        data[2..6].copy_from_slice(&(file_size as u32).to_le_bytes());
        data[10..14].copy_from_slice(&(pixel_offset as u32).to_le_bytes());

        // Info header
        let info_offset = BMP_FILE_HEADER_SIZE;
        data[info_offset..info_offset + 4].copy_from_slice(&40u32.to_le_bytes());
        data[info_offset + 4..info_offset + 8].copy_from_slice(&(width as i32).to_le_bytes());
        data[info_offset + 8..info_offset + 12].copy_from_slice(&(height as i32).to_le_bytes());
        data[info_offset + 12..info_offset + 14].copy_from_slice(&1u16.to_le_bytes());
        data[info_offset + 14..info_offset + 16].copy_from_slice(&24u16.to_le_bytes());

        // Fill with test pattern (BGR)
        for y in 0..height as usize {
            for x in 0..width as usize {
                let offset = pixel_offset + y * row_stride + x * 3;
                data[offset] = (x % 256) as u8; // B
                data[offset + 1] = (y % 256) as u8; // G
                data[offset + 2] = 128; // R
            }
        }

        data
    }

    #[test]
    fn test_decode_small_bmp() {
        let bmp = create_test_bmp(4, 4);
        let frame = decode_bmp(&bmp).unwrap();

        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
        assert!(frame.is_valid());
        assert_eq!(frame.pixels.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_decode_realistic_size() {
        let bmp = create_test_bmp(640, 360);
        let frame = decode_bmp(&bmp).unwrap();

        assert_eq!(frame.width, 640);
        assert_eq!(frame.height, 360);
        assert!(frame.is_valid());
        assert_eq!(frame.pixels.len(), 640 * 360 * 3);
    }

    #[test]
    fn test_bgr_to_rgb_conversion() {
        let bmp = create_test_bmp(2, 2);
        let frame = decode_bmp(&bmp).unwrap();

        // Bottom-left pixel in BMP (row 0, col 0) becomes top-left in output
        // Original BGR: B=0, G=0, R=128
        // After conversion: R=128, G=0, B=0

        // Due to row flipping, check the pattern
        assert_eq!(frame.pixels[0], 128); // R (was at +2 in BGR)
    }

    #[test]
    fn test_bmp_dimensions() {
        let bmp = create_test_bmp(640, 360);
        let dims = bmp_dimensions(&bmp);
        assert_eq!(dims, Some((640, 360)));
    }

    #[test]
    fn test_is_valid_bmp() {
        assert!(is_valid_bmp(b"BM..."));
        assert!(!is_valid_bmp(b"PNG"));
        assert!(!is_valid_bmp(b"B"));
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = BmpDecoder::new(320, 180);
        assert_eq!(decoder.target_dimensions(), (320, 180));
        assert!(decoder.is_resize_enabled());

        let passthrough = BmpDecoder::passthrough();
        assert!(!passthrough.is_resize_enabled());

        let enfusion = BmpDecoder::enfusion_standard();
        assert_eq!(enfusion.target_dimensions(), (640, 360));
    }

    #[test]
    fn test_invalid_signature() {
        let mut data = vec![0u8; 100];
        data[0] = b'X';
        data[1] = b'X';

        let result = decode_bmp(&data);
        assert!(matches!(result, Err(IoError::InvalidBmpHeader(_))));
    }

    #[test]
    fn test_buffer_too_small() {
        let data = vec![b'B', b'M'];
        let result = decode_bmp(&data);
        assert!(matches!(result, Err(IoError::BufferTooSmall { .. })));
    }

    #[test]
    fn test_row_padding() {
        // Width of 5 pixels * 3 bytes = 15 bytes, needs padding to 16
        let bmp = create_test_bmp(5, 3);
        let frame = decode_bmp(&bmp).unwrap();

        assert_eq!(frame.width, 5);
        assert_eq!(frame.height, 3);
        assert!(frame.is_valid());
    }
}
