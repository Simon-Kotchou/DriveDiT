//! CSV telemetry parsing for DriveDiT data pipeline.
//!
//! Provides efficient parsing of telemetry CSV files with frame-indexed lookup.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use drivedit_types::Telemetry;

use crate::Result;

/// Parse telemetry CSV file and return a map from frame_id to (timestamp, telemetry).
pub fn parse_telemetry_csv(path: &Path) -> Result<HashMap<u64, (f64, Telemetry)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut telemetry_map = HashMap::new();

    let mut lines = reader.lines();

    // Skip header line
    if let Some(Ok(_header)) = lines.next() {
        // Parse remaining lines
    }

    let mut frame_id: u64 = 0;
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        // Parse CSV line - minimal parsing
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 5 {
            continue;
        }

        // Try to extract telemetry values
        let timestamp: f64 = fields.first().and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let speed_kmh: f32 = fields.get(1).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let steering: f32 = fields.get(2).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let throttle: f32 = fields.get(3).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
        let brake: f32 = fields.get(4).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);

        let telemetry = Telemetry {
            speed_kmh,
            steering,
            throttle,
            brake,
            ..Default::default()
        };

        telemetry_map.insert(frame_id, (timestamp, telemetry));
        frame_id += 1;
    }

    Ok(telemetry_map)
}

/// Incremental telemetry parser with frame-indexed lookup.
pub struct TelemetryParser {
    /// Path to the CSV file.
    path: std::path::PathBuf,
    /// Parsed telemetry data indexed by frame_id.
    data: HashMap<u64, (f64, Telemetry)>,
    /// Number of lines parsed so far.
    lines_parsed: usize,
}

impl TelemetryParser {
    /// Create a new telemetry parser for the given CSV file.
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.to_path_buf(),
            data: HashMap::new(),
            lines_parsed: 0,
        }
    }

    /// Parse any new updates from the CSV file.
    pub fn parse_updates(&mut self) -> Result<usize> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut count = 0;

        for (idx, line) in reader.lines().enumerate() {
            if idx <= self.lines_parsed {
                continue;
            }

            let line = line?;
            if line.trim().is_empty() {
                continue;
            }

            let fields: Vec<&str> = line.split(',').collect();
            if fields.len() < 5 {
                continue;
            }

            let timestamp: f64 = fields.first().and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
            let speed_kmh: f32 = fields.get(1).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
            let steering: f32 = fields.get(2).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
            let throttle: f32 = fields.get(3).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);
            let brake: f32 = fields.get(4).and_then(|s| s.trim().parse().ok()).unwrap_or(0.0);

            let frame_id = idx as u64;
            let telemetry = Telemetry {
                speed_kmh,
                steering,
                throttle,
                brake,
                ..Default::default()
            };

            self.data.insert(frame_id, (timestamp, telemetry));
            count += 1;
        }

        self.lines_parsed = self.lines_parsed + count;
        Ok(count)
    }

    /// Get telemetry for a specific frame ID.
    pub fn get(&self, frame_id: u64) -> Option<&(f64, Telemetry)> {
        self.data.get(&frame_id)
    }

    /// Get the number of parsed records.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the parser has any data.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get all parsed data.
    pub fn data(&self) -> &HashMap<u64, (f64, Telemetry)> {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_parse_telemetry_csv() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "timestamp,speed,steering,throttle,brake").unwrap();
        writeln!(file, "0.0,60.0,0.1,0.5,0.0").unwrap();
        writeln!(file, "0.1,62.0,0.2,0.6,0.0").unwrap();

        let result = parse_telemetry_csv(file.path()).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_telemetry_parser() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "timestamp,speed,steering,throttle,brake").unwrap();
        writeln!(file, "0.0,60.0,0.1,0.5,0.0").unwrap();

        let mut parser = TelemetryParser::new(file.path());
        let count = parser.parse_updates().unwrap();
        assert!(count > 0);
        assert!(!parser.is_empty());
    }
}
