//! ENFCAP binary format reader/writer for DriveDiT data pipeline.
//!
//! ENFCAP (ENFusion CAPture) is a high-performance binary format for
//! storing captured frames and telemetry from Arma Reforger.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use bitflags::bitflags;
use bytemuck::{Pod, Zeroable};

use crate::Result;
use drivedit_types::Telemetry;

/// ENFCAP file magic bytes.
pub const MAGIC: [u8; 4] = [0x45, 0x4E, 0x46, 0x43]; // "ENFC"

/// Current format version.
pub const VERSION: u32 = 1;

bitflags! {
    /// Header flags for ENFCAP format.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct HeaderFlags: u32 {
        /// Frames are compressed.
        const COMPRESSED = 0b0001;
        /// Contains telemetry data.
        const HAS_TELEMETRY = 0b0010;
        /// Contains depth data.
        const HAS_DEPTH = 0b0100;
        /// Contains anchor frame markers.
        const HAS_ANCHORS = 0b1000;
    }
}

/// ENFCAP file header.
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Header {
    /// Magic bytes ("ENFC").
    pub magic: [u8; 4],
    /// Format version.
    pub version: u32,
    /// Number of frames.
    pub frame_count: u32,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frames per second.
    pub fps: f32,
    /// Flags (compression, etc).
    pub flags: u32,
    /// Reserved for future use.
    pub _reserved: [u8; 28],
}

/// Frame data from ENFCAP file.
#[derive(Debug, Clone)]
pub struct FrameData {
    /// Frame index.
    pub index: u32,
    /// Timestamp in milliseconds.
    pub timestamp_ms: f64,
    /// Raw pixel data (may be compressed).
    pub pixels: Vec<u8>,
    /// Associated telemetry (if present).
    pub telemetry: Option<Telemetry>,
}

/// ENFCAP file reader.
pub struct ENFCAPReader {
    /// File reader.
    reader: BufReader<File>,
    /// Parsed header.
    header: Header,
    /// Frame index table (offset, size).
    index_table: Vec<(u64, u32)>,
}

impl ENFCAPReader {
    /// Open an ENFCAP file for reading.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header_bytes = [0u8; std::mem::size_of::<Header>()];
        reader.read_exact(&mut header_bytes)?;
        let header: Header = bytemuck::cast(header_bytes);

        // Validate magic
        if header.magic != MAGIC {
            return Err(crate::IoError::InvalidHeader("Invalid ENFCAP magic".into()));
        }

        // Read index table
        let mut index_table = Vec::with_capacity(header.frame_count as usize);
        for _ in 0..header.frame_count {
            let mut offset_bytes = [0u8; 8];
            let mut size_bytes = [0u8; 4];
            reader.read_exact(&mut offset_bytes)?;
            reader.read_exact(&mut size_bytes)?;
            let offset = u64::from_le_bytes(offset_bytes);
            let size = u32::from_le_bytes(size_bytes);
            index_table.push((offset, size));
        }

        Ok(Self {
            reader,
            header,
            index_table,
        })
    }

    /// Get the file header.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Get the number of frames.
    pub fn frame_count(&self) -> u32 {
        self.header.frame_count
    }

    /// Get frame dimensions.
    pub fn dimensions(&self) -> (u32, u32) {
        (self.header.width, self.header.height)
    }

    /// Read a specific frame by index.
    pub fn read_frame(&mut self, index: u32) -> Result<FrameData> {
        if index >= self.header.frame_count {
            return Err(crate::IoError::IndexOutOfBounds {
                index: index as usize,
                max: self.header.frame_count as usize,
            });
        }

        let (offset, size) = self.index_table[index as usize];
        self.reader.seek(SeekFrom::Start(offset))?;

        let mut pixels = vec![0u8; size as usize];
        self.reader.read_exact(&mut pixels)?;

        Ok(FrameData {
            index,
            timestamp_ms: (index as f64) * (1000.0 / self.header.fps as f64),
            pixels,
            telemetry: None,
        })
    }

    /// Check if the file has telemetry data.
    pub fn has_telemetry(&self) -> bool {
        let flags = HeaderFlags::from_bits_truncate(self.header.flags);
        flags.contains(HeaderFlags::HAS_TELEMETRY)
    }
}

/// ENFCAP file writer.
pub struct ENFCAPWriter {
    /// File path.
    path: std::path::PathBuf,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Frames per second.
    fps: f32,
    /// Flags.
    flags: HeaderFlags,
    /// Frame data buffer.
    frames: Vec<Vec<u8>>,
}

impl ENFCAPWriter {
    /// Create a new ENFCAP writer.
    pub fn new(path: &Path, width: u32, height: u32, fps: f32) -> Self {
        Self {
            path: path.to_path_buf(),
            width,
            height,
            fps,
            flags: HeaderFlags::empty(),
            frames: Vec::new(),
        }
    }

    /// Add telemetry flag.
    pub fn with_telemetry(mut self) -> Self {
        self.flags |= HeaderFlags::HAS_TELEMETRY;
        self
    }

    /// Add a frame.
    pub fn add_frame(&mut self, pixels: Vec<u8>) {
        self.frames.push(pixels);
    }

    /// Write the ENFCAP file.
    pub fn write(self) -> Result<()> {
        use std::io::Write;

        let mut file = File::create(&self.path)?;

        // Write header
        let header = Header {
            magic: MAGIC,
            version: VERSION,
            frame_count: self.frames.len() as u32,
            width: self.width,
            height: self.height,
            fps: self.fps,
            flags: self.flags.bits(),
            _reserved: [0; 28],
        };
        file.write_all(bytemuck::bytes_of(&header))?;

        // Calculate offsets and write index table
        let header_size = std::mem::size_of::<Header>() as u64;
        let index_entry_size = 12u64; // 8 bytes offset + 4 bytes size
        let index_table_size = index_entry_size * self.frames.len() as u64;
        let mut current_offset = header_size + index_table_size;

        for frame in &self.frames {
            file.write_all(&current_offset.to_le_bytes())?;
            file.write_all(&(frame.len() as u32).to_le_bytes())?;
            current_offset += frame.len() as u64;
        }

        // Write frame data
        for frame in &self.frames {
            file.write_all(frame)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_enfcap_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let path = temp_dir.path().join("test.enfcap");

        // Write
        let mut writer = ENFCAPWriter::new(&path, 640, 360, 30.0);
        writer.add_frame(vec![0u8; 640 * 360 * 3]);
        writer.add_frame(vec![1u8; 640 * 360 * 3]);
        writer.write().unwrap();

        // Read
        let mut reader = ENFCAPReader::open(&path).unwrap();
        assert_eq!(reader.frame_count(), 2);
        assert_eq!(reader.dimensions(), (640, 360));

        let frame0 = reader.read_frame(0).unwrap();
        assert_eq!(frame0.index, 0);
        assert_eq!(frame0.pixels.len(), 640 * 360 * 3);
    }
}
