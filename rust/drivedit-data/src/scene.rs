//! High-performance scene graph parsing with SIMD-JSON.
//!
//! This module provides efficient scene graph loading for DriveDiT training pipelines,
//! with support for:
//! - Fast JSON parsing using simd-json
//! - Entity representation with 3D bounding boxes
//! - Velocity and trajectory tracking
//! - Minimal allocations during parsing

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

use nalgebra::{Matrix3, Point3, Quaternion, UnitQuaternion, Vector3};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during scene parsing
#[derive(Error, Debug)]
pub enum SceneError {
    #[error("Failed to parse scene from {path}: {message}")]
    ParseError { path: String, message: String },

    #[error("Invalid entity data: {0}")]
    InvalidEntity(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(String),
}

/// 3D bounding box representation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoundingBox3D {
    /// Center position (x, y, z) in world coordinates
    pub center: [f32; 3],
    /// Dimensions (length, width, height)
    pub dimensions: [f32; 3],
    /// Rotation as quaternion (w, x, y, z)
    pub rotation: [f32; 4],
}

impl Default for BoundingBox3D {
    fn default() -> Self {
        Self {
            center: [0.0; 3],
            dimensions: [1.0, 1.0, 1.0],
            rotation: [1.0, 0.0, 0.0, 0.0], // Identity quaternion
        }
    }
}

impl BoundingBox3D {
    /// Create a new bounding box
    pub fn new(center: [f32; 3], dimensions: [f32; 3], rotation: [f32; 4]) -> Self {
        Self {
            center,
            dimensions,
            rotation,
        }
    }

    /// Create from center and dimensions (axis-aligned)
    pub fn axis_aligned(center: [f32; 3], dimensions: [f32; 3]) -> Self {
        Self {
            center,
            dimensions,
            rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// Get the 8 corner points of the bounding box
    pub fn corners(&self) -> [[f32; 3]; 8] {
        let (l, w, h) = (
            self.dimensions[0] / 2.0,
            self.dimensions[1] / 2.0,
            self.dimensions[2] / 2.0,
        );

        // Local corners before rotation
        let local_corners = [
            [-l, -w, -h],
            [l, -w, -h],
            [l, w, -h],
            [-l, w, -h],
            [-l, -w, h],
            [l, -w, h],
            [l, w, h],
            [-l, w, h],
        ];

        // Create rotation quaternion
        let q = UnitQuaternion::from_quaternion(Quaternion::new(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        ));

        let center = Vector3::new(self.center[0], self.center[1], self.center[2]);

        let mut corners = [[0.0f32; 3]; 8];
        for (i, local) in local_corners.iter().enumerate() {
            let local_vec = Vector3::new(local[0], local[1], local[2]);
            let rotated = q * local_vec + center;
            corners[i] = [rotated.x, rotated.y, rotated.z];
        }

        corners
    }

    /// Get the rotation matrix
    pub fn rotation_matrix(&self) -> [[f32; 3]; 3] {
        let q = UnitQuaternion::from_quaternion(Quaternion::new(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        ));
        let m = q.to_rotation_matrix();
        let m = m.matrix();
        [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)]],
        ]
    }

    /// Get volume of the bounding box
    #[inline]
    pub fn volume(&self) -> f32 {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }

    /// Check if a point is inside the bounding box (accounting for rotation)
    pub fn contains(&self, point: [f32; 3]) -> bool {
        // Transform point to local coordinates
        let q = UnitQuaternion::from_quaternion(Quaternion::new(
            self.rotation[0],
            self.rotation[1],
            self.rotation[2],
            self.rotation[3],
        ));
        let inv_q = q.inverse();

        let center = Vector3::new(self.center[0], self.center[1], self.center[2]);
        let p = Vector3::new(point[0], point[1], point[2]);
        let local = inv_q * (p - center);

        let (hl, hw, hh) = (
            self.dimensions[0] / 2.0,
            self.dimensions[1] / 2.0,
            self.dimensions[2] / 2.0,
        );

        local.x.abs() <= hl && local.y.abs() <= hw && local.z.abs() <= hh
    }

    /// Compute IoU (Intersection over Union) with another bounding box
    /// Approximation using axis-aligned bounds
    pub fn iou_approx(&self, other: &BoundingBox3D) -> f32 {
        // Get axis-aligned bounds
        let (min_a, max_a) = self.axis_aligned_bounds();
        let (min_b, max_b) = other.axis_aligned_bounds();

        // Intersection
        let inter_min = [
            min_a[0].max(min_b[0]),
            min_a[1].max(min_b[1]),
            min_a[2].max(min_b[2]),
        ];
        let inter_max = [
            max_a[0].min(max_b[0]),
            max_a[1].min(max_b[1]),
            max_a[2].min(max_b[2]),
        ];

        let inter_vol = (inter_max[0] - inter_min[0]).max(0.0)
            * (inter_max[1] - inter_min[1]).max(0.0)
            * (inter_max[2] - inter_min[2]).max(0.0);

        let union_vol = self.volume() + other.volume() - inter_vol;

        if union_vol > 0.0 {
            inter_vol / union_vol
        } else {
            0.0
        }
    }

    /// Get axis-aligned bounding box (min, max corners)
    fn axis_aligned_bounds(&self) -> ([f32; 3], [f32; 3]) {
        let corners = self.corners();
        let mut min = corners[0];
        let mut max = corners[0];

        for corner in &corners[1..] {
            for i in 0..3 {
                min[i] = min[i].min(corner[i]);
                max[i] = max[i].max(corner[i]);
            }
        }

        (min, max)
    }
}

/// Entity class/type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EntityClass {
    Vehicle,
    Pedestrian,
    Cyclist,
    Motorcycle,
    Truck,
    Bus,
    TrafficLight,
    TrafficSign,
    Barrier,
    Cone,
    Other,
    #[serde(other)]
    Unknown,
}

impl Default for EntityClass {
    fn default() -> Self {
        Self::Unknown
    }
}

impl EntityClass {
    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "vehicle" | "car" | "automobile" => Self::Vehicle,
            "pedestrian" | "person" | "human" => Self::Pedestrian,
            "cyclist" | "bicycle" | "bike" => Self::Cyclist,
            "motorcycle" | "motorbike" => Self::Motorcycle,
            "truck" | "lorry" => Self::Truck,
            "bus" => Self::Bus,
            "traffic_light" | "trafficlight" | "light" => Self::TrafficLight,
            "traffic_sign" | "trafficsign" | "sign" => Self::TrafficSign,
            "barrier" | "fence" => Self::Barrier,
            "cone" | "traffic_cone" => Self::Cone,
            "other" => Self::Other,
            _ => Self::Unknown,
        }
    }

    /// Check if this is a dynamic (moving) entity class
    #[inline]
    pub fn is_dynamic(&self) -> bool {
        matches!(
            self,
            Self::Vehicle
                | Self::Pedestrian
                | Self::Cyclist
                | Self::Motorcycle
                | Self::Truck
                | Self::Bus
        )
    }

    /// Check if this is a static entity class
    #[inline]
    pub fn is_static(&self) -> bool {
        !self.is_dynamic()
    }
}

/// An entity in the scene
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier
    pub id: u64,
    /// Entity class
    pub class: EntityClass,
    /// 3D position (x, y, z) in world coordinates
    pub position: [f32; 3],
    /// 3D bounding box
    pub bbox_3d: BoundingBox3D,
    /// Velocity (vx, vy, vz) in m/s
    pub velocity: [f32; 3],
    /// Optional 2D bounding box in image coordinates (x, y, w, h)
    #[serde(default)]
    pub bbox_2d: Option<[f32; 4]>,
    /// Confidence score (0-1)
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// Track ID for temporal association
    #[serde(default)]
    pub track_id: Option<u64>,
    /// Optional attributes (e.g., "occluded", "truncated")
    #[serde(default)]
    pub attributes: HashMap<String, String>,
}

fn default_confidence() -> f32 {
    1.0
}

impl Default for Entity {
    fn default() -> Self {
        Self {
            id: 0,
            class: EntityClass::Unknown,
            position: [0.0; 3],
            bbox_3d: BoundingBox3D::default(),
            velocity: [0.0; 3],
            bbox_2d: None,
            confidence: 1.0,
            track_id: None,
            attributes: HashMap::new(),
        }
    }
}

impl Entity {
    /// Create a new entity
    pub fn new(id: u64, class: EntityClass, position: [f32; 3], bbox_3d: BoundingBox3D) -> Self {
        Self {
            id,
            class,
            position,
            bbox_3d,
            ..Default::default()
        }
    }

    /// Compute speed (magnitude of velocity)
    #[inline]
    pub fn speed(&self) -> f32 {
        let v = &self.velocity;
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    /// Check if entity is moving
    #[inline]
    pub fn is_moving(&self, threshold: f32) -> bool {
        self.speed() > threshold
    }

    /// Get distance from origin
    #[inline]
    pub fn distance_from_origin(&self) -> f32 {
        let p = &self.position;
        (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt()
    }

    /// Get distance to another entity
    #[inline]
    pub fn distance_to(&self, other: &Entity) -> f32 {
        let dx = self.position[0] - other.position[0];
        let dy = self.position[1] - other.position[1];
        let dz = self.position[2] - other.position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Predict position at future time (constant velocity model)
    pub fn predict_position(&self, dt: f32) -> [f32; 3] {
        [
            self.position[0] + self.velocity[0] * dt,
            self.position[1] + self.velocity[1] * dt,
            self.position[2] + self.velocity[2] * dt,
        ]
    }
}

/// A complete scene graph for one frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneGraph {
    /// Frame/timestamp identifier
    pub frame_id: u64,
    /// Timestamp in seconds
    pub timestamp: f64,
    /// List of entities in the scene
    pub entities: Vec<Entity>,
    /// Ego vehicle pose (optional)
    #[serde(default)]
    pub ego_pose: Option<EgoPose>,
    /// Additional metadata
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// Ego vehicle pose
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct EgoPose {
    /// Position (x, y, z) in world coordinates
    pub position: [f32; 3],
    /// Rotation as quaternion (w, x, y, z)
    pub rotation: [f32; 4],
    /// Velocity (vx, vy, vz) in m/s
    pub velocity: [f32; 3],
    /// Angular velocity (wx, wy, wz) in rad/s
    #[serde(default)]
    pub angular_velocity: [f32; 3],
}

impl Default for EgoPose {
    fn default() -> Self {
        Self {
            position: [0.0; 3],
            rotation: [1.0, 0.0, 0.0, 0.0],
            velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
        }
    }
}

impl SceneGraph {
    /// Create an empty scene graph
    pub fn new(frame_id: u64, timestamp: f64) -> Self {
        Self {
            frame_id,
            timestamp,
            entities: Vec::new(),
            ego_pose: None,
            metadata: HashMap::new(),
        }
    }

    /// Add an entity to the scene
    pub fn add_entity(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Get entity by ID
    pub fn get_entity(&self, id: u64) -> Option<&Entity> {
        self.entities.iter().find(|e| e.id == id)
    }

    /// Get entities by class
    pub fn get_by_class(&self, class: EntityClass) -> Vec<&Entity> {
        self.entities.iter().filter(|e| e.class == class).collect()
    }

    /// Get dynamic entities
    pub fn get_dynamic(&self) -> Vec<&Entity> {
        self.entities.iter().filter(|e| e.class.is_dynamic()).collect()
    }

    /// Get static entities
    pub fn get_static(&self) -> Vec<&Entity> {
        self.entities.iter().filter(|e| e.class.is_static()).collect()
    }

    /// Get entities within distance from a point
    pub fn get_within_distance(&self, point: [f32; 3], max_distance: f32) -> Vec<&Entity> {
        let max_dist_sq = max_distance * max_distance;
        self.entities
            .iter()
            .filter(|e| {
                let dx = e.position[0] - point[0];
                let dy = e.position[1] - point[1];
                let dz = e.position[2] - point[2];
                dx * dx + dy * dy + dz * dz <= max_dist_sq
            })
            .collect()
    }

    /// Get entities within ego vehicle's field of view
    pub fn get_in_fov(&self, fov_deg: f32, max_range: f32) -> Vec<&Entity> {
        let fov_rad = fov_deg.to_radians() / 2.0;

        self.entities
            .iter()
            .filter(|e| {
                let x = e.position[0];
                let y = e.position[1];
                let dist = (x * x + y * y).sqrt();

                if dist > max_range {
                    return false;
                }

                // Assume ego facing +x direction
                let angle = y.atan2(x).abs();
                angle <= fov_rad
            })
            .collect()
    }

    /// Number of entities
    #[inline]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Check if scene is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Get entity class histogram
    pub fn class_histogram(&self) -> HashMap<EntityClass, usize> {
        let mut histogram = HashMap::new();
        for entity in &self.entities {
            *histogram.entry(entity.class).or_insert(0) += 1;
        }
        histogram
    }
}

/// High-performance scene parser using simd-json
pub struct SceneParser {
    /// Buffer for JSON parsing (reused to minimize allocations)
    buffer: Vec<u8>,
}

impl SceneParser {
    /// Create a new scene parser
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(65536), // 64KB initial capacity
        }
    }

    /// Parse scene from a file
    pub fn parse_file<P: AsRef<Path>>(&mut self, path: P) -> Result<SceneGraph, SceneError> {
        let path = path.as_ref();
        let mut file = File::open(path)?;

        self.buffer.clear();
        file.read_to_end(&mut self.buffer)?;

        self.parse_bytes(&self.buffer.clone())
    }

    /// Parse scene from bytes using simd-json
    pub fn parse_bytes(&mut self, bytes: &[u8]) -> Result<SceneGraph, SceneError> {
        // simd-json requires mutable bytes
        let mut bytes_copy = bytes.to_vec();

        let scene: SceneGraph = simd_json::from_slice(&mut bytes_copy)
            .map_err(|e| SceneError::JsonError(e.to_string()))?;

        Ok(scene)
    }

    /// Parse scene from a JSON string
    pub fn parse_str(&mut self, json: &str) -> Result<SceneGraph, SceneError> {
        self.parse_bytes(json.as_bytes())
    }

    /// Parse multiple scene files in parallel
    pub fn parse_batch<P: AsRef<Path> + Sync>(
        &self,
        paths: &[P],
    ) -> Vec<Result<SceneGraph, SceneError>> {
        paths
            .par_iter()
            .map(|path| {
                let mut parser = SceneParser::new();
                parser.parse_file(path)
            })
            .collect()
    }

    /// Parse multiple scenes, returning only successful parses
    pub fn parse_batch_ok<P: AsRef<Path> + Sync>(&self, paths: &[P]) -> Vec<SceneGraph> {
        paths
            .par_iter()
            .filter_map(|path| {
                let mut parser = SceneParser::new();
                parser.parse_file(path).ok()
            })
            .collect()
    }
}

impl Default for SceneParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Utilities for scene graph processing
pub mod utils {
    use super::*;

    /// Match entities between two frames using Hungarian algorithm approximation
    pub fn match_entities(
        prev: &SceneGraph,
        curr: &SceneGraph,
        max_distance: f32,
    ) -> Vec<(u64, u64)> {
        let mut matches = Vec::new();
        let mut used_curr: Vec<bool> = vec![false; curr.entities.len()];

        // Simple greedy matching (not optimal but fast)
        for prev_entity in &prev.entities {
            let mut best_match: Option<(usize, f32)> = None;

            for (i, curr_entity) in curr.entities.iter().enumerate() {
                if used_curr[i] {
                    continue;
                }
                if prev_entity.class != curr_entity.class {
                    continue;
                }

                let dist = prev_entity.distance_to(curr_entity);
                if dist > max_distance {
                    continue;
                }

                match best_match {
                    None => best_match = Some((i, dist)),
                    Some((_, best_dist)) if dist < best_dist => {
                        best_match = Some((i, dist));
                    }
                    _ => {}
                }
            }

            if let Some((i, _)) = best_match {
                matches.push((prev_entity.id, curr.entities[i].id));
                used_curr[i] = true;
            }
        }

        matches
    }

    /// Compute entity velocities from position differences
    pub fn compute_velocities(
        prev: &SceneGraph,
        curr: &mut SceneGraph,
        matches: &[(u64, u64)],
    ) {
        let dt = (curr.timestamp - prev.timestamp) as f32;
        if dt <= 0.0 {
            return;
        }

        let prev_map: HashMap<u64, &Entity> =
            prev.entities.iter().map(|e| (e.id, e)).collect();

        for (prev_id, curr_id) in matches {
            if let Some(prev_entity) = prev_map.get(prev_id) {
                if let Some(curr_entity) = curr.entities.iter_mut().find(|e| e.id == *curr_id) {
                    let inv_dt = 1.0 / dt;
                    curr_entity.velocity = [
                        (curr_entity.position[0] - prev_entity.position[0]) * inv_dt,
                        (curr_entity.position[1] - prev_entity.position[1]) * inv_dt,
                        (curr_entity.position[2] - prev_entity.position[2]) * inv_dt,
                    ];
                }
            }
        }
    }

    /// Transform scene to ego vehicle frame
    pub fn transform_to_ego(scene: &SceneGraph) -> SceneGraph {
        let mut transformed = scene.clone();

        if let Some(ego) = &scene.ego_pose {
            let ego_pos = Vector3::new(ego.position[0], ego.position[1], ego.position[2]);
            let ego_rot = UnitQuaternion::from_quaternion(Quaternion::new(
                ego.rotation[0],
                ego.rotation[1],
                ego.rotation[2],
                ego.rotation[3],
            ));
            let inv_rot = ego_rot.inverse();

            for entity in &mut transformed.entities {
                // Transform position
                let pos = Vector3::new(
                    entity.position[0],
                    entity.position[1],
                    entity.position[2],
                );
                let local_pos = inv_rot * (pos - ego_pos);
                entity.position = [local_pos.x, local_pos.y, local_pos.z];

                // Transform velocity
                let vel = Vector3::new(
                    entity.velocity[0],
                    entity.velocity[1],
                    entity.velocity[2],
                );
                let local_vel = inv_rot * vel;
                entity.velocity = [local_vel.x, local_vel.y, local_vel.z];

                // Transform bounding box center
                let bbox_center = Vector3::new(
                    entity.bbox_3d.center[0],
                    entity.bbox_3d.center[1],
                    entity.bbox_3d.center[2],
                );
                let local_center = inv_rot * (bbox_center - ego_pos);
                entity.bbox_3d.center = [local_center.x, local_center.y, local_center.z];

                // Transform bounding box rotation
                let bbox_rot = UnitQuaternion::from_quaternion(Quaternion::new(
                    entity.bbox_3d.rotation[0],
                    entity.bbox_3d.rotation[1],
                    entity.bbox_3d.rotation[2],
                    entity.bbox_3d.rotation[3],
                ));
                let local_rot = inv_rot * bbox_rot;
                let q = local_rot.quaternion();
                entity.bbox_3d.rotation = [q.w, q.i, q.j, q.k];
            }

            // Reset ego pose to origin
            transformed.ego_pose = Some(EgoPose::default());
        }

        transformed
    }

    /// Filter entities by confidence threshold
    pub fn filter_by_confidence(scene: &mut SceneGraph, min_confidence: f32) {
        scene.entities.retain(|e| e.confidence >= min_confidence);
    }

    /// Convert scene to feature vector for neural network input
    /// Returns [N, F] array where N is number of entities and F is feature dimension
    pub fn to_feature_vector(
        scene: &SceneGraph,
        max_entities: usize,
        feature_dim: usize,
    ) -> Vec<f32> {
        let mut features = vec![0.0f32; max_entities * feature_dim];

        for (i, entity) in scene.entities.iter().take(max_entities).enumerate() {
            let offset = i * feature_dim;

            // Position (3)
            features[offset] = entity.position[0];
            features[offset + 1] = entity.position[1];
            features[offset + 2] = entity.position[2];

            // Velocity (3)
            features[offset + 3] = entity.velocity[0];
            features[offset + 4] = entity.velocity[1];
            features[offset + 5] = entity.velocity[2];

            // Bounding box dimensions (3)
            features[offset + 6] = entity.bbox_3d.dimensions[0];
            features[offset + 7] = entity.bbox_3d.dimensions[1];
            features[offset + 8] = entity.bbox_3d.dimensions[2];

            // Class (one-hot, assuming 11 classes)
            if feature_dim > 9 {
                let class_idx = entity.class as usize;
                if offset + 9 + class_idx < features.len() {
                    features[offset + 9 + class_idx] = 1.0;
                }
            }
        }

        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounding_box_corners() {
        let bbox = BoundingBox3D::axis_aligned([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
        let corners = bbox.corners();

        // Check that corners are at correct positions
        assert_eq!(corners[0], [-1.0, -1.0, -1.0]);
        assert_eq!(corners[6], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_bounding_box_contains() {
        let bbox = BoundingBox3D::axis_aligned([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);

        assert!(bbox.contains([0.0, 0.0, 0.0]));
        assert!(bbox.contains([0.5, 0.5, 0.5]));
        assert!(!bbox.contains([2.0, 0.0, 0.0]));
    }

    #[test]
    fn test_entity_class_parsing() {
        assert_eq!(EntityClass::from_str("vehicle"), EntityClass::Vehicle);
        assert_eq!(EntityClass::from_str("Car"), EntityClass::Vehicle);
        assert_eq!(EntityClass::from_str("PEDESTRIAN"), EntityClass::Pedestrian);
        assert_eq!(EntityClass::from_str("unknown_type"), EntityClass::Unknown);
    }

    #[test]
    fn test_entity_speed() {
        let mut entity = Entity::default();
        entity.velocity = [3.0, 4.0, 0.0];
        assert!((entity.speed() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_scene_graph_queries() {
        let mut scene = SceneGraph::new(0, 0.0);

        scene.add_entity(Entity::new(
            1,
            EntityClass::Vehicle,
            [10.0, 0.0, 0.0],
            BoundingBox3D::default(),
        ));
        scene.add_entity(Entity::new(
            2,
            EntityClass::Pedestrian,
            [5.0, 0.0, 0.0],
            BoundingBox3D::default(),
        ));
        scene.add_entity(Entity::new(
            3,
            EntityClass::Vehicle,
            [100.0, 0.0, 0.0],
            BoundingBox3D::default(),
        ));

        assert_eq!(scene.len(), 3);
        assert_eq!(scene.get_by_class(EntityClass::Vehicle).len(), 2);
        assert_eq!(scene.get_within_distance([0.0, 0.0, 0.0], 20.0).len(), 2);
    }

    #[test]
    fn test_json_parsing() {
        let json = r#"{
            "frame_id": 42,
            "timestamp": 1.5,
            "entities": [
                {
                    "id": 1,
                    "class": "vehicle",
                    "position": [10.0, 5.0, 0.0],
                    "bbox_3d": {
                        "center": [10.0, 5.0, 0.5],
                        "dimensions": [4.5, 2.0, 1.5],
                        "rotation": [1.0, 0.0, 0.0, 0.0]
                    },
                    "velocity": [5.0, 0.0, 0.0]
                }
            ]
        }"#;

        let mut parser = SceneParser::new();
        let scene = parser.parse_str(json).unwrap();

        assert_eq!(scene.frame_id, 42);
        assert_eq!(scene.entities.len(), 1);
        assert_eq!(scene.entities[0].class, EntityClass::Vehicle);
    }
}
