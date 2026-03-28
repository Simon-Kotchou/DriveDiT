//! JPEG and PNG image decoding with auto-format detection.
//!
//! This module provides high-performance image decoding using the `image` crate,
//! with support for JPEG, PNG, and automatic format detection.
//!
//! # Example
//!
//! ```rust,ignore
//! use drivedit_io::jpeg::{decode_jpeg, decode_png, decode_auto};
//!
//! // Decode JPEG
//! let frame = decode_jpeg(&jpeg_bytes)?;
//!
//! // Decode PNG
//! let frame = decode_png(&png_bytes)?;
//!
//! // Auto-detect format
//! let frame = decode_auto(&unknown_bytes)?;
//! ```

use crate::error::{IoError, Result};
use drivedit_types::{RawFrame, Telemetry};
use image::{GenericImageView, ImageFormat, ImageReader};
use std::io::Cursor;
use std::path::Path;

/// Decode a JPEG image from raw bytes.
///
/// Returns a [`RawFrame`] with RGB pixel data in row-major [H, W, C] layout.
pub fn decode_jpeg(data: &[u8]) -> Result<RawFrame> {
    let cursor = Cursor::new(data);
    let img = ImageReader::with_format(cursor, ImageFormat::Jpeg)
        .decode()
        .map_err(|e| IoError::ImageDecode(format!("JPEG decode error: {}", e)))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    Ok(RawFrame::new(0, 0.0, pixels, width, height, Telemetry::default()))
}

/// Decode a PNG image from raw bytes.
///
/// Returns a [`RawFrame`] with RGB pixel data in row-major [H, W, C] layout.
pub fn decode_png(data: &[u8]) -> Result<RawFrame> {
    let cursor = Cursor::new(data);
    let img = ImageReader::with_format(cursor, ImageFormat::Png)
        .decode()
        .map_err(|e| IoError::ImageDecode(format!("PNG decode error: {}", e)))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    Ok(RawFrame::new(0, 0.0, pixels, width, height, Telemetry::default()))
}

/// Auto-detect image format and decode.
///
/// Supports BMP, JPEG, PNG, and other formats recognized by the `image` crate.
pub fn decode_auto(data: &[u8]) -> Result<RawFrame> {
    let cursor = Cursor::new(data);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| IoError::ImageDecode(format!("Format detection error: {}", e)))?
        .decode()
        .map_err(|e| IoError::ImageDecode(format!("Decode error: {}", e)))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    Ok(RawFrame::new(0, 0.0, pixels, width, height, Telemetry::default()))
}

/// Decode an image file with automatic format detection.
pub fn decode_file(path: &Path) -> Result<RawFrame> {
    let img = ImageReader::open(path)
        .map_err(|e| IoError::FileIo(e.into()))?
        .decode()
        .map_err(|e| IoError::ImageDecode(format!("Decode error: {}", e)))?;

    let rgb = img.to_rgb8();
    let (width, height) = rgb.dimensions();
    let pixels = rgb.into_raw();

    Ok(RawFrame::new(0, 0.0, pixels, width, height, Telemetry::default()))
}

/// Decode and resize an image from bytes.
pub fn decode_and_resize(
    data: &[u8],
    target_width: u32,
    target_height: u32,
) -> Result<RawFrame> {
    let cursor = Cursor::new(data);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| IoError::ImageDecode(format!("Format detection error: {}", e)))?
        .decode()
        .map_err(|e| IoError::ImageDecode(format!("Decode error: {}", e)))?;

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

/// Detect image format from magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageType {
    /// JPEG image (starts with 0xFF 0xD8)
    Jpeg,
    /// PNG image (starts with PNG signature)
    Png,
    /// BMP image (starts with "BM")
    Bmp,
    /// WebP image
    WebP,
    /// Unknown format
    Unknown,
}

impl ImageType {
    /// Detect image type from the first few bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        if data.len() < 4 {
            return ImageType::Unknown;
        }

        // JPEG: starts with 0xFF 0xD8 0xFF
        if data[0] == 0xFF && data[1] == 0xD8 && data[2] == 0xFF {
            return ImageType::Jpeg;
        }

        // PNG: starts with 0x89 PNG \r\n
        if data[0] == 0x89
            && data[1] == b'P'
            && data[2] == b'N'
            && data[3] == b'G'
        {
            return ImageType::Png;
        }

        // BMP: starts with "BM"
        if data[0] == b'B' && data[1] == b'M' {
            return ImageType::Bmp;
        }

        // WebP: starts with "RIFF" and contains "WEBP"
        if data.len() >= 12
            && &data[0..4] == b"RIFF"
            && &data[8..12] == b"WEBP"
        {
            return ImageType::WebP;
        }

        ImageType::Unknown
    }
}

/// Check if a path is an image file based on extension.
pub fn is_image_file(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            matches!(
                e.to_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "bmp" | "webp"
            )
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_type_detection() {
        // JPEG magic bytes
        let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        assert_eq!(ImageType::from_bytes(&jpeg_data), ImageType::Jpeg);

        // PNG magic bytes
        let png_data = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(ImageType::from_bytes(&png_data), ImageType::Png);

        // BMP magic bytes
        let bmp_data = [b'B', b'M', 0x00, 0x00];
        assert_eq!(ImageType::from_bytes(&bmp_data), ImageType::Bmp);

        // Unknown
        let unknown_data = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(ImageType::from_bytes(&unknown_data), ImageType::Unknown);
    }

    #[test]
    fn test_is_image_file() {
        assert!(is_image_file(Path::new("test.jpg")));
        assert!(is_image_file(Path::new("test.JPEG")));
        assert!(is_image_file(Path::new("test.png")));
        assert!(is_image_file(Path::new("test.PNG")));
        assert!(is_image_file(Path::new("test.bmp")));
        assert!(is_image_file(Path::new("test.webp")));
        assert!(!is_image_file(Path::new("test.txt")));
        assert!(!is_image_file(Path::new("test")));
    }

    // Note: Full decode tests require actual image data, which we don't have in unit tests.
    // Integration tests with real image files should be added separately.
}
