//! Telemetry Processing Module
//!
//! High-performance telemetry data parsing and processing for autonomous driving
//! training data. Includes:
//! - Fast CSV parsing with typed columns
//! - SIMD-accelerated control signal normalization
//! - Ego transform computation with nalgebra
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use drivedit_data::telemetry::{TelemetryParser, ControlNormalizer, EgoTransformComputer};
//!
//! // Parse telemetry CSV
//! let parser = TelemetryParser::new();
//! let records = parser.parse_file("telemetry.csv")?;
//!
//! // Normalize control signals for training
//! let normalizer = ControlNormalizer::default();
//! let normalized = normalizer.normalize_batch(&records);
//!
//! // Compute ego transforms
//! let computer = EgoTransformComputer::new();
//! let transforms = computer.compute_batch(&records);
//! ```

use csv::ReaderBuilder;
use nalgebra::{Matrix3, Matrix4, Rotation3, Unit, Vector3};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use thiserror::Error;

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during telemetry processing
#[derive(Error, Debug)]
pub enum TelemetryError {
    #[error("CSV parsing error: {0}")]
    CsvError(#[from] csv::Error),

    #[error("Missing required column: {0}")]
    MissingColumn(String),

    #[error("Invalid value in column {column}: {value}")]
    InvalidValue { column: String, value: String },

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),
}

pub type Result<T> = std::result::Result<T, TelemetryError>;

// ============================================================================
// TelemetryRecord - Complete Telemetry Row
// ============================================================================

/// Complete telemetry record for a single frame
///
/// Contains all sensor readings and vehicle state for one timestamp.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TelemetryRecord {
    /// Timestamp in milliseconds since session start
    pub timestamp_ms: f64,

    /// Frame ID (if synchronized with video)
    pub frame_id: Option<u32>,

    // --- Position and Orientation ---
    /// GPS latitude in degrees
    pub latitude: f64,
    /// GPS longitude in degrees
    pub longitude: f64,
    /// Altitude in meters
    pub altitude: f32,
    /// Heading in radians (0 = North, positive = clockwise)
    pub heading: f32,
    /// Pitch angle in radians
    pub pitch: f32,
    /// Roll angle in radians
    pub roll: f32,

    // --- Vehicle Dynamics ---
    /// Speed in km/h
    pub speed_kmh: f32,
    /// Speed in m/s (computed or direct)
    pub speed_ms: f32,
    /// Longitudinal acceleration in m/s^2
    pub acceleration_x: f32,
    /// Lateral acceleration in m/s^2
    pub acceleration_y: f32,
    /// Vertical acceleration in m/s^2
    pub acceleration_z: f32,
    /// Yaw rate in rad/s
    pub yaw_rate: f32,
    /// Slip angle in radians
    pub slip_angle: f32,

    // --- Control Inputs ---
    /// Steering wheel angle (-1.0 to 1.0 normalized, or raw degrees)
    pub steering: f32,
    /// Throttle position (0.0 to 1.0)
    pub throttle: f32,
    /// Brake position (0.0 to 1.0)
    pub brake: f32,
    /// Clutch position (0.0 to 1.0)
    pub clutch: f32,

    // --- Vehicle State ---
    /// Current gear (-1 = reverse, 0 = neutral, 1+ = forward)
    pub gear: i8,
    /// Engine RPM
    pub engine_rpm: f32,
    /// Fuel level (0.0 to 1.0)
    pub fuel_level: f32,
    /// Engine temperature in Celsius
    pub engine_temp: f32,

    // --- Indicators ---
    /// Turn signal state (-1 = left, 0 = none, 1 = right)
    pub turn_signal: i8,
    /// Headlights on
    pub headlights: bool,
    /// Wipers on
    pub wipers: bool,
    /// Hazard lights on
    pub hazards: bool,

    // --- Quality Flags ---
    /// GPS fix quality (0-3, higher is better)
    pub gps_quality: u8,
    /// Record is valid
    pub is_valid: bool,
}

impl TelemetryRecord {
    /// Convert speed from km/h to m/s
    #[inline]
    pub fn compute_speed_ms(&mut self) {
        self.speed_ms = self.speed_kmh / 3.6;
    }

    /// Get the total acceleration magnitude
    #[inline]
    pub fn total_acceleration(&self) -> f32 {
        (self.acceleration_x.powi(2) + self.acceleration_y.powi(2) + self.acceleration_z.powi(2))
            .sqrt()
    }

    /// Check if the vehicle is stationary
    #[inline]
    pub fn is_stationary(&self) -> bool {
        self.speed_kmh < 0.5
    }

    /// Check if braking
    #[inline]
    pub fn is_braking(&self) -> bool {
        self.brake > 0.05
    }

    /// Check if accelerating
    #[inline]
    pub fn is_accelerating(&self) -> bool {
        self.throttle > 0.05 && self.brake < 0.05
    }

    /// Get normalized control vector [steering, throttle, brake]
    pub fn control_vector(&self) -> [f32; 3] {
        [self.steering, self.throttle, self.brake]
    }

    /// Get position as [lat, lon, alt]
    pub fn position(&self) -> [f64; 3] {
        [self.latitude, self.longitude, self.altitude as f64]
    }

    /// Get orientation as [heading, pitch, roll]
    pub fn orientation(&self) -> [f32; 3] {
        [self.heading, self.pitch, self.roll]
    }
}

// ============================================================================
// TelemetryParser - Fast CSV Parsing
// ============================================================================

/// Column mapping configuration for CSV parsing
#[derive(Debug, Clone)]
pub struct ColumnMapping {
    /// Map from our field names to CSV column names
    pub mappings: HashMap<String, String>,
}

impl Default for ColumnMapping {
    fn default() -> Self {
        let mut mappings = HashMap::new();

        // Common column name variations
        mappings.insert("timestamp_ms".into(), "timestamp_ms".into());
        mappings.insert("frame_id".into(), "frame_id".into());
        mappings.insert("latitude".into(), "lat".into());
        mappings.insert("longitude".into(), "lng".into());
        mappings.insert("altitude".into(), "alt".into());
        mappings.insert("heading".into(), "heading".into());
        mappings.insert("pitch".into(), "pitch".into());
        mappings.insert("roll".into(), "roll".into());
        mappings.insert("speed_kmh".into(), "speed_kmh".into());
        mappings.insert("steering".into(), "steering".into());
        mappings.insert("throttle".into(), "throttle".into());
        mappings.insert("brake".into(), "brake".into());
        mappings.insert("gear".into(), "gear".into());
        mappings.insert("engine_rpm".into(), "rpm".into());
        mappings.insert("yaw_rate".into(), "yaw_rate".into());
        mappings.insert("acceleration_x".into(), "accel_x".into());
        mappings.insert("acceleration_y".into(), "accel_y".into());
        mappings.insert("acceleration_z".into(), "accel_z".into());

        Self { mappings }
    }
}

impl ColumnMapping {
    /// Create a new column mapping with custom field names
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    /// Add a column mapping
    pub fn map(&mut self, field: &str, csv_column: &str) -> &mut Self {
        self.mappings.insert(field.into(), csv_column.into());
        self
    }

    /// Get the CSV column name for a field
    pub fn get(&self, field: &str) -> Option<&str> {
        self.mappings.get(field).map(|s| s.as_str())
    }
}

/// High-performance CSV telemetry parser
#[pyclass]
pub struct TelemetryParser {
    /// Column mapping configuration
    column_mapping: ColumnMapping,
    /// Whether to skip invalid records
    skip_invalid: bool,
    /// Whether to compute derived fields
    compute_derived: bool,
}

impl TelemetryParser {
    /// Create a new parser with default settings
    pub fn new() -> Self {
        Self {
            column_mapping: ColumnMapping::default(),
            skip_invalid: true,
            compute_derived: true,
        }
    }

    /// Create parser with custom column mapping
    pub fn with_mapping(mapping: ColumnMapping) -> Self {
        Self {
            column_mapping: mapping,
            skip_invalid: true,
            compute_derived: true,
        }
    }

    /// Set whether to skip invalid records
    pub fn skip_invalid(mut self, skip: bool) -> Self {
        self.skip_invalid = skip;
        self
    }

    /// Set whether to compute derived fields
    pub fn compute_derived(mut self, compute: bool) -> Self {
        self.compute_derived = compute;
        self
    }

    /// Parse a CSV file into telemetry records
    pub fn parse_file<P: AsRef<Path>>(&self, path: P) -> Result<Vec<TelemetryRecord>> {
        let file = File::open(path)?;
        let reader = BufReader::with_capacity(256 * 1024, file);
        self.parse_reader(reader)
    }

    /// Parse from a reader
    pub fn parse_reader<R: Read>(&self, reader: R) -> Result<Vec<TelemetryRecord>> {
        let mut csv_reader = ReaderBuilder::new()
            .flexible(true)
            .has_headers(true)
            .from_reader(reader);

        let headers = csv_reader.headers()?.clone();
        let column_indices = self.build_column_indices(&headers)?;

        let mut records = Vec::with_capacity(10000);

        for result in csv_reader.records() {
            match result {
                Ok(record) => {
                    match self.parse_record(&record, &column_indices) {
                        Ok(mut telemetry) => {
                            if self.compute_derived {
                                telemetry.compute_speed_ms();
                            }
                            telemetry.is_valid = true;
                            records.push(telemetry);
                        }
                        Err(_e) => {
                            if !self.skip_invalid {
                                return Err(_e);
                            }
                            // Skip invalid record
                        }
                    }
                }
                Err(e) => {
                    if !self.skip_invalid {
                        return Err(e.into());
                    }
                }
            }
        }

        Ok(records)
    }

    /// Parse CSV data from bytes
    pub fn parse_bytes(&self, data: &[u8]) -> Result<Vec<TelemetryRecord>> {
        self.parse_reader(data)
    }

    /// Build column index map for fast access
    fn build_column_indices(
        &self,
        headers: &csv::StringRecord,
    ) -> Result<HashMap<String, usize>> {
        let mut indices = HashMap::new();

        for (idx, header) in headers.iter().enumerate() {
            let header_lower = header.to_lowercase().trim().to_string();
            indices.insert(header_lower, idx);
        }

        Ok(indices)
    }

    /// Parse a single CSV record
    fn parse_record(
        &self,
        record: &csv::StringRecord,
        column_indices: &HashMap<String, usize>,
    ) -> Result<TelemetryRecord> {
        let mut telemetry = TelemetryRecord::default();

        // Helper function to parse with multiple possible column names
        fn try_parse_f64(
            record: &csv::StringRecord,
            indices: &HashMap<String, usize>,
            names: &[&str],
        ) -> Option<f64> {
            for name in names {
                if let Some(&idx) = indices.get(&name.to_lowercase()) {
                    if let Some(val) = record.get(idx) {
                        if let Ok(parsed) = val.trim().parse::<f64>() {
                            return Some(parsed);
                        }
                    }
                }
            }
            None
        }

        fn try_parse_f32(
            record: &csv::StringRecord,
            indices: &HashMap<String, usize>,
            names: &[&str],
        ) -> Option<f32> {
            for name in names {
                if let Some(&idx) = indices.get(&name.to_lowercase()) {
                    if let Some(val) = record.get(idx) {
                        if let Ok(parsed) = val.trim().parse::<f32>() {
                            return Some(parsed);
                        }
                    }
                }
            }
            None
        }

        fn try_parse_i8(
            record: &csv::StringRecord,
            indices: &HashMap<String, usize>,
            names: &[&str],
        ) -> Option<i8> {
            for name in names {
                if let Some(&idx) = indices.get(&name.to_lowercase()) {
                    if let Some(val) = record.get(idx) {
                        if let Ok(parsed) = val.trim().parse::<i8>() {
                            return Some(parsed);
                        }
                    }
                }
            }
            None
        }

        fn try_parse_u32(
            record: &csv::StringRecord,
            indices: &HashMap<String, usize>,
            names: &[&str],
        ) -> Option<u32> {
            for name in names {
                if let Some(&idx) = indices.get(&name.to_lowercase()) {
                    if let Some(val) = record.get(idx) {
                        if let Ok(parsed) = val.trim().parse::<u32>() {
                            return Some(parsed);
                        }
                    }
                }
            }
            None
        }

        // Parse timestamp (required)
        if let Some(ts) = try_parse_f64(
            record,
            column_indices,
            &["timestamp_ms", "timestamp", "time_ms", "time"],
        ) {
            telemetry.timestamp_ms = ts;
        }

        // Parse frame ID
        telemetry.frame_id =
            try_parse_u32(record, column_indices, &["frame_id", "frame", "id"]);

        // Parse position
        if let Some(v) = try_parse_f64(record, column_indices, &["latitude", "lat"]) {
            telemetry.latitude = v;
        }
        if let Some(v) = try_parse_f64(record, column_indices, &["longitude", "lng", "lon"]) {
            telemetry.longitude = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["altitude", "alt"]) {
            telemetry.altitude = v;
        }

        // Parse orientation
        if let Some(v) = try_parse_f32(record, column_indices, &["heading", "yaw", "bearing"]) {
            telemetry.heading = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["pitch"]) {
            telemetry.pitch = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["roll"]) {
            telemetry.roll = v;
        }

        // Parse speed
        if let Some(v) =
            try_parse_f32(record, column_indices, &["speed_kmh", "speed", "velocity_kmh"])
        {
            telemetry.speed_kmh = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["speed_ms", "velocity_ms"]) {
            telemetry.speed_ms = v;
        }

        // Parse controls
        if let Some(v) =
            try_parse_f32(record, column_indices, &["steering", "steer", "steering_angle"])
        {
            telemetry.steering = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["throttle", "gas", "accel"]) {
            telemetry.throttle = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["brake", "braking"]) {
            telemetry.brake = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["clutch"]) {
            telemetry.clutch = v;
        }

        // Parse vehicle state
        if let Some(v) = try_parse_i8(record, column_indices, &["gear"]) {
            telemetry.gear = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["engine_rpm", "rpm"]) {
            telemetry.engine_rpm = v;
        }

        // Parse accelerations
        if let Some(v) =
            try_parse_f32(record, column_indices, &["acceleration_x", "accel_x", "ax"])
        {
            telemetry.acceleration_x = v;
        }
        if let Some(v) =
            try_parse_f32(record, column_indices, &["acceleration_y", "accel_y", "ay"])
        {
            telemetry.acceleration_y = v;
        }
        if let Some(v) =
            try_parse_f32(record, column_indices, &["acceleration_z", "accel_z", "az"])
        {
            telemetry.acceleration_z = v;
        }
        if let Some(v) = try_parse_f32(record, column_indices, &["yaw_rate", "gyro_z"]) {
            telemetry.yaw_rate = v;
        }

        Ok(telemetry)
    }

    /// Parse multiple files in parallel
    pub fn parse_files_parallel<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Result<Vec<Vec<TelemetryRecord>>> {
        paths.par_iter().map(|p| self.parse_file(p)).collect()
    }
}

impl Default for TelemetryParser {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ControlNormalizer - SIMD-Accelerated Normalization
// ============================================================================

/// Control signal normalization configuration
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// Steering range [min, max] in raw units
    pub steering_range: (f32, f32),
    /// Whether steering input is in degrees
    pub steering_degrees: bool,
    /// Maximum steering angle in degrees (if steering_degrees is true)
    pub max_steering_degrees: f32,
    /// Throttle range [min, max]
    pub throttle_range: (f32, f32),
    /// Brake range [min, max]
    pub brake_range: (f32, f32),
    /// Output range for all controls
    pub output_range: (f32, f32),
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self {
            steering_range: (-1.0, 1.0),
            steering_degrees: false,
            max_steering_degrees: 450.0,
            throttle_range: (0.0, 1.0),
            brake_range: (0.0, 1.0),
            output_range: (-1.0, 1.0),
        }
    }
}

/// Normalized control vector for ML training
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(16))]
pub struct NormalizedControls {
    /// Steering (-1.0 = full left, 1.0 = full right)
    pub steering: f32,
    /// Throttle (0.0 to 1.0)
    pub throttle: f32,
    /// Brake (0.0 to 1.0)
    pub brake: f32,
    /// Combined throttle-brake (-1.0 = full brake, 1.0 = full throttle)
    pub accel_brake: f32,
}

impl NormalizedControls {
    /// Convert to array for neural network input
    #[inline]
    pub fn to_array(&self) -> [f32; 4] {
        [self.steering, self.throttle, self.brake, self.accel_brake]
    }

    /// Create from array
    #[inline]
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self {
            steering: arr[0],
            throttle: arr[1],
            brake: arr[2],
            accel_brake: arr[3],
        }
    }
}

/// SIMD-accelerated control signal normalizer
pub struct ControlNormalizer {
    config: NormalizationConfig,
}

impl ControlNormalizer {
    /// Create a new normalizer with custom config
    pub fn new(config: NormalizationConfig) -> Self {
        Self { config }
    }

    /// Normalize a single control record
    #[inline]
    pub fn normalize(&self, record: &TelemetryRecord) -> NormalizedControls {
        let steering = self.normalize_steering(record.steering);
        let throttle = self.normalize_throttle(record.throttle);
        let brake = self.normalize_brake(record.brake);

        // Combined accel-brake: throttle - brake, clamped to [-1, 1]
        let accel_brake = (throttle - brake).clamp(-1.0, 1.0);

        NormalizedControls {
            steering,
            throttle,
            brake,
            accel_brake,
        }
    }

    /// Normalize steering value
    #[inline]
    fn normalize_steering(&self, value: f32) -> f32 {
        let (min, max) = self.config.steering_range;
        let normalized = if self.config.steering_degrees {
            value / self.config.max_steering_degrees
        } else {
            (value - min) / (max - min) * 2.0 - 1.0
        };
        normalized.clamp(-1.0, 1.0)
    }

    /// Normalize throttle value
    #[inline]
    fn normalize_throttle(&self, value: f32) -> f32 {
        let (min, max) = self.config.throttle_range;
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Normalize brake value
    #[inline]
    fn normalize_brake(&self, value: f32) -> f32 {
        let (min, max) = self.config.brake_range;
        ((value - min) / (max - min)).clamp(0.0, 1.0)
    }

    /// Batch normalize records (vectorized when possible)
    pub fn normalize_batch(&self, records: &[TelemetryRecord]) -> Vec<NormalizedControls> {
        let mut results = Vec::with_capacity(records.len());

        // Process in chunks for better cache performance
        for chunk in records.chunks(8) {
            let batch_results = self.normalize_batch_8(chunk);
            results.extend(batch_results);
        }

        results
    }

    /// Normalize a batch of up to 8 records (optimized for SIMD-friendly processing)
    #[inline]
    fn normalize_batch_8(&self, records: &[TelemetryRecord]) -> Vec<NormalizedControls> {
        let (s_min, s_max) = self.config.steering_range;
        let s_range = s_max - s_min;
        let (t_min, t_max) = self.config.throttle_range;
        let t_range = t_max - t_min;
        let (b_min, b_max) = self.config.brake_range;
        let b_range = b_max - b_min;

        records
            .iter()
            .map(|record| {
                // Normalize steering
                let normalized_steering = if self.config.steering_degrees {
                    record.steering / self.config.max_steering_degrees
                } else {
                    (record.steering - s_min) / s_range * 2.0 - 1.0
                };
                let steering = normalized_steering.clamp(-1.0, 1.0);

                // Normalize throttle
                let throttle = ((record.throttle - t_min) / t_range).clamp(0.0, 1.0);

                // Normalize brake
                let brake = ((record.brake - b_min) / b_range).clamp(0.0, 1.0);

                // Compute accel-brake
                let accel_brake = (throttle - brake).clamp(-1.0, 1.0);

                NormalizedControls {
                    steering,
                    throttle,
                    brake,
                    accel_brake,
                }
            })
            .collect()
    }

    /// Batch normalize in parallel (for large datasets)
    pub fn normalize_batch_parallel(&self, records: &[TelemetryRecord]) -> Vec<NormalizedControls> {
        const CHUNK_SIZE: usize = 1024;

        if records.len() < CHUNK_SIZE * 2 {
            return self.normalize_batch(records);
        }

        records
            .par_chunks(CHUNK_SIZE)
            .flat_map(|chunk| self.normalize_batch(chunk))
            .collect()
    }
}

impl Default for ControlNormalizer {
    fn default() -> Self {
        Self::new(NormalizationConfig::default())
    }
}

// ============================================================================
// EgoTransformComputer - Matrix Computation with nalgebra
// ============================================================================

/// Computed ego vehicle transformation
#[derive(Debug, Clone)]
pub struct EgoTransform {
    /// 4x4 homogeneous transformation matrix
    pub matrix: Matrix4<f32>,
    /// Position in world coordinates (x, y, z)
    pub position: Vector3<f32>,
    /// Rotation matrix (3x3)
    pub rotation: Matrix3<f32>,
    /// Forward direction unit vector
    pub forward: Vector3<f32>,
    /// Right direction unit vector
    pub right: Vector3<f32>,
    /// Up direction unit vector
    pub up: Vector3<f32>,
}

impl EgoTransform {
    /// Create identity transform
    pub fn identity() -> Self {
        Self {
            matrix: Matrix4::identity(),
            position: Vector3::zeros(),
            rotation: Matrix3::identity(),
            forward: Vector3::new(0.0, 0.0, 1.0),
            right: Vector3::new(1.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
        }
    }

    /// Get the inverse transformation
    pub fn inverse(&self) -> Option<Self> {
        self.matrix.try_inverse().map(|inv_matrix| {
            let inv_rotation = self.rotation.transpose();
            Self {
                matrix: inv_matrix,
                position: -inv_rotation * self.position,
                rotation: inv_rotation,
                forward: inv_rotation * self.forward,
                right: inv_rotation * self.right,
                up: inv_rotation * self.up,
            }
        })
    }

    /// Transform a point from ego to world coordinates
    #[inline]
    pub fn transform_point(&self, point: &Vector3<f32>) -> Vector3<f32> {
        self.rotation * point + self.position
    }

    /// Transform a direction from ego to world coordinates
    #[inline]
    pub fn transform_direction(&self, direction: &Vector3<f32>) -> Vector3<f32> {
        self.rotation * direction
    }

    /// Convert to row-major 3x4 array (compatible with ENFCAP format)
    pub fn to_array_3x4(&self) -> [f32; 12] {
        [
            self.right.x,
            self.right.y,
            self.right.z,
            self.up.x,
            self.up.y,
            self.up.z,
            self.forward.x,
            self.forward.y,
            self.forward.z,
            self.position.x,
            self.position.y,
            self.position.z,
        ]
    }

    /// Create from row-major 3x4 array
    pub fn from_array_3x4(arr: &[f32; 12]) -> Self {
        let right = Vector3::new(arr[0], arr[1], arr[2]);
        let up = Vector3::new(arr[3], arr[4], arr[5]);
        let forward = Vector3::new(arr[6], arr[7], arr[8]);
        let position = Vector3::new(arr[9], arr[10], arr[11]);

        let rotation = Matrix3::from_columns(&[right, up, forward]);

        let matrix = Matrix4::new(
            rotation[(0, 0)],
            rotation[(0, 1)],
            rotation[(0, 2)],
            position.x,
            rotation[(1, 0)],
            rotation[(1, 1)],
            rotation[(1, 2)],
            position.y,
            rotation[(2, 0)],
            rotation[(2, 1)],
            rotation[(2, 2)],
            position.z,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        Self {
            matrix,
            position,
            rotation,
            forward,
            right,
            up,
        }
    }
}

impl Default for EgoTransform {
    fn default() -> Self {
        Self::identity()
    }
}

/// Configuration for ego transform computation
#[derive(Debug, Clone)]
pub struct TransformConfig {
    /// Earth radius for GPS to local conversion (meters)
    pub earth_radius: f64,
    /// Reference latitude for local coordinate frame
    pub ref_latitude: f64,
    /// Reference longitude for local coordinate frame
    pub ref_longitude: f64,
    /// Reference altitude for local coordinate frame
    pub ref_altitude: f32,
    /// Whether to use reference point from first record
    pub auto_reference: bool,
}

impl Default for TransformConfig {
    fn default() -> Self {
        Self {
            earth_radius: 6_371_000.0, // meters
            ref_latitude: 0.0,
            ref_longitude: 0.0,
            ref_altitude: 0.0,
            auto_reference: true,
        }
    }
}

/// Ego transform computer with GPS to local coordinate conversion
pub struct EgoTransformComputer {
    config: TransformConfig,
    /// Cached reference position (set from first record if auto_reference)
    ref_set: bool,
}

impl EgoTransformComputer {
    /// Create a new transform computer
    pub fn new() -> Self {
        Self {
            config: TransformConfig::default(),
            ref_set: false,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: TransformConfig) -> Self {
        let ref_set = !config.auto_reference;
        Self { config, ref_set }
    }

    /// Set reference point manually
    pub fn set_reference(&mut self, lat: f64, lon: f64, alt: f32) {
        self.config.ref_latitude = lat;
        self.config.ref_longitude = lon;
        self.config.ref_altitude = alt;
        self.ref_set = true;
    }

    /// Compute transform for a single telemetry record
    pub fn compute(&mut self, record: &TelemetryRecord) -> EgoTransform {
        // Set reference from first record if auto_reference
        if !self.ref_set && self.config.auto_reference {
            self.config.ref_latitude = record.latitude;
            self.config.ref_longitude = record.longitude;
            self.config.ref_altitude = record.altitude;
            self.ref_set = true;
        }

        // Convert GPS to local coordinates
        let position = self.gps_to_local(record.latitude, record.longitude, record.altitude);

        // Build rotation from heading, pitch, roll
        let rotation = self.build_rotation(record.heading, record.pitch, record.roll);

        // Extract basis vectors
        let right = rotation.column(0).into();
        let up = rotation.column(1).into();
        let forward = rotation.column(2).into();

        // Build 4x4 homogeneous matrix
        let matrix = Matrix4::new(
            rotation[(0, 0)],
            rotation[(0, 1)],
            rotation[(0, 2)],
            position.x,
            rotation[(1, 0)],
            rotation[(1, 1)],
            rotation[(1, 2)],
            position.y,
            rotation[(2, 0)],
            rotation[(2, 1)],
            rotation[(2, 2)],
            position.z,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        EgoTransform {
            matrix,
            position,
            rotation,
            forward,
            right,
            up,
        }
    }

    /// Compute transforms for a batch of records
    pub fn compute_batch(&mut self, records: &[TelemetryRecord]) -> Vec<EgoTransform> {
        records.iter().map(|r| self.compute(r)).collect()
    }

    /// Compute transforms in parallel (requires pre-set reference)
    pub fn compute_batch_parallel(&self, records: &[TelemetryRecord]) -> Vec<EgoTransform> {
        assert!(
            self.ref_set,
            "Reference must be set for parallel computation"
        );

        records
            .par_iter()
            .map(|record| {
                let position =
                    self.gps_to_local(record.latitude, record.longitude, record.altitude);
                let rotation = self.build_rotation(record.heading, record.pitch, record.roll);

                let right = rotation.column(0).into();
                let up = rotation.column(1).into();
                let forward = rotation.column(2).into();

                let matrix = Matrix4::new(
                    rotation[(0, 0)],
                    rotation[(0, 1)],
                    rotation[(0, 2)],
                    position.x,
                    rotation[(1, 0)],
                    rotation[(1, 1)],
                    rotation[(1, 2)],
                    position.y,
                    rotation[(2, 0)],
                    rotation[(2, 1)],
                    rotation[(2, 2)],
                    position.z,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                );

                EgoTransform {
                    matrix,
                    position,
                    rotation,
                    forward,
                    right,
                    up,
                }
            })
            .collect()
    }

    /// Convert GPS coordinates to local ENU (East-North-Up) coordinates
    fn gps_to_local(&self, lat: f64, lon: f64, alt: f32) -> Vector3<f32> {
        let lat_rad = lat.to_radians();
        let lon_rad = lon.to_radians();
        let ref_lat_rad = self.config.ref_latitude.to_radians();
        let ref_lon_rad = self.config.ref_longitude.to_radians();

        // Approximate conversion using equirectangular projection
        let d_lon = lon_rad - ref_lon_rad;
        let d_lat = lat_rad - ref_lat_rad;

        let x = (d_lon * ref_lat_rad.cos() * self.config.earth_radius) as f32;
        let y = (d_lat * self.config.earth_radius) as f32;
        let z = alt - self.config.ref_altitude;

        Vector3::new(x, z, y) // Reorder for typical 3D coordinate system
    }

    /// Build rotation matrix from heading, pitch, roll (ZYX convention)
    fn build_rotation(&self, heading: f32, pitch: f32, roll: f32) -> Matrix3<f32> {
        // Convert heading to standard rotation (heading 0 = North = +Z)
        let yaw = -heading; // Negate for right-hand rule

        // Build rotation using nalgebra
        let rotation_yaw = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), yaw);
        let rotation_pitch = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::x()), pitch);
        let rotation_roll = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::z()), roll);

        // ZYX order: roll * pitch * yaw
        let rotation = rotation_roll * rotation_pitch * rotation_yaw;
        *rotation.matrix()
    }

    /// Compute relative transform between two frames
    pub fn compute_relative(&self, from: &EgoTransform, to: &EgoTransform) -> Option<EgoTransform> {
        from.inverse().map(|inv| {
            let relative_matrix = inv.matrix * to.matrix;
            let relative_rotation = inv.rotation * to.rotation;
            let relative_position = inv.rotation * (to.position - from.position);

            EgoTransform {
                matrix: relative_matrix,
                position: relative_position,
                rotation: relative_rotation,
                forward: relative_rotation.column(2).into(),
                right: relative_rotation.column(0).into(),
                up: relative_rotation.column(1).into(),
            }
        })
    }

    /// Compute velocity from consecutive transforms
    pub fn compute_velocity(
        &self,
        prev: &EgoTransform,
        curr: &EgoTransform,
        dt_seconds: f32,
    ) -> Vector3<f32> {
        if dt_seconds > 0.0 {
            (curr.position - prev.position) / dt_seconds
        } else {
            Vector3::zeros()
        }
    }
}

impl Default for EgoTransformComputer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Trajectory Processing
// ============================================================================

/// A sequence of ego transforms representing a trajectory
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// Ordered list of transforms
    pub transforms: Vec<EgoTransform>,
    /// Corresponding timestamps
    pub timestamps: Vec<f64>,
}

impl Trajectory {
    /// Create a new empty trajectory
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Create trajectory from telemetry records
    pub fn from_records(records: &[TelemetryRecord]) -> Self {
        let mut computer = EgoTransformComputer::new();
        let transforms = computer.compute_batch(records);
        let timestamps: Vec<f64> = records.iter().map(|r| r.timestamp_ms).collect();

        Self {
            transforms,
            timestamps,
        }
    }

    /// Get the total distance traveled
    pub fn total_distance(&self) -> f32 {
        self.transforms
            .windows(2)
            .map(|w| (w[1].position - w[0].position).magnitude())
            .sum()
    }

    /// Get the total duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        if self.timestamps.len() < 2 {
            return 0.0;
        }
        self.timestamps.last().unwrap() - self.timestamps.first().unwrap()
    }

    /// Get average speed in m/s
    pub fn average_speed(&self) -> f32 {
        let duration_s = self.duration_ms() / 1000.0;
        if duration_s > 0.0 {
            self.total_distance() / duration_s as f32
        } else {
            0.0
        }
    }

    /// Interpolate transform at a given timestamp
    pub fn interpolate(&self, timestamp_ms: f64) -> Option<EgoTransform> {
        if self.timestamps.is_empty() {
            return None;
        }

        // Find surrounding timestamps
        let idx = self
            .timestamps
            .iter()
            .position(|&t| t > timestamp_ms)
            .unwrap_or(self.timestamps.len());

        if idx == 0 {
            return Some(self.transforms[0].clone());
        }
        if idx >= self.timestamps.len() {
            return Some(self.transforms.last()?.clone());
        }

        // Linear interpolation
        let t0 = self.timestamps[idx - 1];
        let t1 = self.timestamps[idx];
        let alpha = ((timestamp_ms - t0) / (t1 - t0)) as f32;

        let pos0 = &self.transforms[idx - 1].position;
        let pos1 = &self.transforms[idx].position;
        let position = pos0.lerp(pos1, alpha);

        // Use the closest rotation (simple approach - could use slerp)
        let rotation = if alpha < 0.5 {
            self.transforms[idx - 1].rotation
        } else {
            self.transforms[idx].rotation
        };

        Some(EgoTransform {
            matrix: Matrix4::identity(), // Would need proper interpolation
            position,
            rotation,
            forward: rotation.column(2).into(),
            right: rotation.column(0).into(),
            up: rotation.column(1).into(),
        })
    }
}

impl Default for Trajectory {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_record_defaults() {
        let record = TelemetryRecord::default();
        assert_eq!(record.speed_kmh, 0.0);
        assert!(!record.is_valid);
        assert!(record.is_stationary());
    }

    #[test]
    fn test_control_normalization() {
        let normalizer = ControlNormalizer::default();

        let record = TelemetryRecord {
            steering: 0.5,
            throttle: 0.8,
            brake: 0.0,
            ..Default::default()
        };

        let normalized = normalizer.normalize(&record);
        assert!((normalized.steering - 0.5).abs() < 0.01);
        assert!((normalized.throttle - 0.8).abs() < 0.01);
        assert!((normalized.brake - 0.0).abs() < 0.01);
        assert!((normalized.accel_brake - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_batch_normalization() {
        let normalizer = ControlNormalizer::default();

        let records: Vec<TelemetryRecord> = (0..16)
            .map(|i| TelemetryRecord {
                steering: (i as f32) / 15.0 * 2.0 - 1.0,
                throttle: (i as f32) / 15.0,
                brake: 0.0,
                ..Default::default()
            })
            .collect();

        let normalized = normalizer.normalize_batch(&records);
        assert_eq!(normalized.len(), 16);
    }

    #[test]
    fn test_ego_transform_computation() {
        let mut computer = EgoTransformComputer::new();

        let record = TelemetryRecord {
            latitude: 37.7749,
            longitude: -122.4194,
            altitude: 10.0,
            heading: 0.0,
            pitch: 0.0,
            roll: 0.0,
            ..Default::default()
        };

        let transform = computer.compute(&record);

        // First record should be at origin
        assert!(transform.position.magnitude() < 0.01);
    }

    #[test]
    fn test_transform_array_roundtrip() {
        let original = EgoTransform {
            matrix: Matrix4::identity(),
            position: Vector3::new(10.0, 20.0, 30.0),
            rotation: Matrix3::identity(),
            forward: Vector3::new(0.0, 0.0, 1.0),
            right: Vector3::new(1.0, 0.0, 0.0),
            up: Vector3::new(0.0, 1.0, 0.0),
        };

        let arr = original.to_array_3x4();
        let restored = EgoTransform::from_array_3x4(&arr);

        assert!((restored.position - original.position).magnitude() < 0.001);
    }

    #[test]
    fn test_trajectory_distance() {
        let records: Vec<TelemetryRecord> = (0..10)
            .map(|i| TelemetryRecord {
                timestamp_ms: i as f64 * 100.0,
                latitude: 37.7749 + (i as f64) * 0.0001,
                longitude: -122.4194,
                altitude: 0.0,
                heading: 0.0,
                ..Default::default()
            })
            .collect();

        let trajectory = Trajectory::from_records(&records);
        assert!(trajectory.total_distance() > 0.0);
    }

    #[test]
    fn test_column_mapping() {
        let mut mapping = ColumnMapping::new();
        mapping.map("speed_kmh", "vehicle_speed");
        mapping.map("steering", "wheel_angle");

        assert_eq!(mapping.get("speed_kmh"), Some("vehicle_speed"));
        assert_eq!(mapping.get("steering"), Some("wheel_angle"));
        assert_eq!(mapping.get("unknown"), None);
    }
}
