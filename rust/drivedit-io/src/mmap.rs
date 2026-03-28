//! Memory-mapped file utilities for DriveDiT data pipeline.
//!
//! Provides efficient memory-mapped access to large data files.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::Result;

/// Memory-mapped file wrapper for efficient I/O.
pub struct MappedFile {
    /// The underlying memory map.
    mmap: Mmap,
    /// File path.
    path: std::path::PathBuf,
}

impl MappedFile {
    /// Open a file and memory-map it.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self {
            mmap,
            path: path.to_path_buf(),
        })
    }

    /// Get the raw bytes of the mapped file.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Get the file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the size of the mapped region.
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Check if the file is empty.
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Read a slice of bytes starting at offset.
    pub fn slice(&self, offset: usize, len: usize) -> Option<&[u8]> {
        if offset + len <= self.mmap.len() {
            Some(&self.mmap[offset..offset + len])
        } else {
            None
        }
    }
}

/// A region within a memory-mapped file.
pub struct MappedRegion<'a> {
    /// The data slice.
    data: &'a [u8],
    /// Offset within the file.
    offset: usize,
}

impl<'a> MappedRegion<'a> {
    /// Create a new region from a file slice.
    pub fn new(data: &'a [u8], offset: usize) -> Self {
        Self { data, offset }
    }

    /// Get the region data.
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Get the offset within the file.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the length of the region.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the region is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mapped_file() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello, World!").unwrap();

        let mapped = MappedFile::open(file.path()).unwrap();
        assert_eq!(mapped.as_bytes(), b"Hello, World!");
        assert_eq!(mapped.len(), 13);
    }

    #[test]
    fn test_mapped_slice() {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(b"Hello, World!").unwrap();

        let mapped = MappedFile::open(file.path()).unwrap();
        assert_eq!(mapped.slice(0, 5), Some(b"Hello".as_slice()));
        assert_eq!(mapped.slice(7, 5), Some(b"World".as_slice()));
        assert_eq!(mapped.slice(0, 100), None); // Out of bounds
    }
}
