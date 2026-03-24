"""
GAIA-2-Style Rich Conditioning Module for World Model

Implements comprehensive conditioning for autonomous driving world modeling:
- Camera intrinsics/extrinsics encoding
- Road topology encoding with cross-attention
- 3D bounding box encoding for detected objects
- Ego vehicle state encoding
- Temporal and environmental conditioning
- AdaLN-based integration with classifier-free guidance support

Reference: GAIA-2 world model architecture for autonomous driving
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
from einops import rearrange, repeat
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Configuration
# =============================================================================

class ObjectClass(Enum):
    """Object class types for 3D bounding box encoding."""
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    MOTORCYCLE = 3
    TRUCK = 4
    BUS = 5
    TRAILER = 6
    CONSTRUCTION = 7
    TRAFFIC_SIGN = 8
    TRAFFIC_LIGHT = 9
    OTHER = 10


class RoadType(Enum):
    """Road type categories."""
    HIGHWAY = 0
    URBAN = 1
    RURAL = 2
    INTERSECTION = 3
    ROUNDABOUT = 4
    PARKING = 5
    OFFRAMP = 6
    ONRAMP = 7


class WeatherCondition(Enum):
    """Weather condition categories."""
    CLEAR = 0
    CLOUDY = 1
    RAIN = 2
    SNOW = 3
    FOG = 4
    NIGHT = 5
    DUSK = 6
    DAWN = 7


@dataclass
class ConditioningConfig:
    """Configuration for conditioning modules."""
    # Model dimensions
    model_dim: int = 512
    cond_dim: int = 256

    # Camera encoding
    num_cameras: int = 6
    camera_embed_dim: int = 128

    # Road topology
    max_waypoints: int = 50
    max_lanes: int = 8
    road_embed_dim: int = 128
    road_cross_attn_heads: int = 4

    # 3D bounding boxes
    max_objects: int = 64
    bbox_embed_dim: int = 128
    num_object_classes: int = 11

    # Ego state
    ego_state_dim: int = 64

    # Temporal/environmental
    time_embed_dim: int = 64
    weather_embed_dim: int = 32

    # Integration
    num_adaln_modulation_outputs: int = 6  # scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp
    dropout: float = 0.1

    # Classifier-free guidance
    cfg_dropout_prob: float = 0.1


# =============================================================================
# Helper Functions
# =============================================================================

def normalize_intrinsics(K: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    """
    Normalize camera intrinsic matrix to [0, 1] range.

    Args:
        K: Intrinsic matrix [B, 3, 3] or [B, N_cameras, 3, 3]
        image_size: (height, width) of the image

    Returns:
        Normalized intrinsic matrix
    """
    H, W = image_size
    K_norm = K.clone()

    if K.dim() == 3:  # [B, 3, 3]
        K_norm[:, 0, 0] /= W  # fx
        K_norm[:, 1, 1] /= H  # fy
        K_norm[:, 0, 2] /= W  # cx
        K_norm[:, 1, 2] /= H  # cy
    elif K.dim() == 4:  # [B, N, 3, 3]
        K_norm[:, :, 0, 0] /= W
        K_norm[:, :, 1, 1] /= H
        K_norm[:, :, 0, 2] /= W
        K_norm[:, :, 1, 2] /= H

    return K_norm


def rotation_matrix_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    Convert 3x3 rotation matrix to 6D representation (first two columns).

    Args:
        R: Rotation matrix [B, 3, 3] or [B, N, 3, 3]

    Returns:
        6D representation [B, 6] or [B, N, 6]
    """
    if R.dim() == 3:  # [B, 3, 3]
        return R[:, :, :2].reshape(-1, 6)
    else:  # [B, N, 3, 3]
        B, N = R.shape[:2]
        return R[:, :, :, :2].reshape(B, N, 6)


def sinusoidal_embedding(x: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """
    Create sinusoidal positional embedding for scalar values.

    Args:
        x: Input values [B] or [B, N]
        dim: Embedding dimension
        max_period: Maximum period for sinusoidal functions

    Returns:
        Sinusoidal embedding [B, dim] or [B, N, dim]
    """
    half_dim = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half_dim, device=x.device, dtype=x.dtype) / half_dim
    )

    if x.dim() == 1:  # [B]
        x = x.unsqueeze(-1)  # [B, 1]
        freqs = freqs.unsqueeze(0)  # [1, half_dim]
        args = x * freqs  # [B, half_dim]
    else:  # [B, N]
        x = x.unsqueeze(-1)  # [B, N, 1]
        freqs = freqs.view(1, 1, -1)  # [1, 1, half_dim]
        args = x * freqs  # [B, N, half_dim]

    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # Handle odd dimensions
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))

    return embedding


# =============================================================================
# Camera Intrinsics/Extrinsics Encoder
# =============================================================================

class CameraEncoder(nn.Module):
    """
    Encode camera intrinsic and extrinsic parameters.

    Intrinsics (K): 3x3 matrix containing fx, fy, cx, cy, skew
    Extrinsics (E): 3x4 matrix [R|t] containing rotation and translation

    Supports multi-camera setups with proper normalization.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        output_dim: int = 256,
        num_cameras: int = 6,
        image_size: Tuple[int, int] = (256, 256),
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.num_cameras = num_cameras
        self.image_size = image_size

        # Intrinsics encoder: 5 values (fx, fy, cx, cy, skew)
        self.intrinsics_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Extrinsics encoder: 6D rotation + 3D translation = 9 values
        self.extrinsics_encoder = nn.Sequential(
            nn.Linear(9, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Camera ID embedding for multi-camera setup
        self.camera_id_embed = nn.Embedding(num_cameras, embed_dim)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, output_dim),
            nn.Dropout(dropout)
        )

        # Aggregation for multi-camera to single vector
        self.aggregation = nn.Sequential(
            nn.Linear(output_dim * num_cameras, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _extract_intrinsics_features(self, K: torch.Tensor) -> torch.Tensor:
        """
        Extract features from intrinsic matrix.

        Args:
            K: Intrinsic matrix [B, N, 3, 3] normalized

        Returns:
            Features [B, N, 5]
        """
        # Extract relevant parameters
        fx = K[:, :, 0, 0]
        fy = K[:, :, 1, 1]
        cx = K[:, :, 0, 2]
        cy = K[:, :, 1, 2]
        skew = K[:, :, 0, 1]

        return torch.stack([fx, fy, cx, cy, skew], dim=-1)

    def _extract_extrinsics_features(self, E: torch.Tensor) -> torch.Tensor:
        """
        Extract features from extrinsic matrix.

        Args:
            E: Extrinsic matrix [B, N, 3, 4]

        Returns:
            Features [B, N, 9] (6D rotation + 3D translation)
        """
        R = E[:, :, :, :3]  # [B, N, 3, 3]
        t = E[:, :, :, 3]   # [B, N, 3]

        # Convert rotation to 6D representation
        rot_6d = rotation_matrix_to_6d(R)  # [B, N, 6]

        # Normalize translation (assuming meters, normalize to reasonable range)
        t_norm = t / 100.0  # Assuming max translation ~100m

        return torch.cat([rot_6d, t_norm], dim=-1)

    def forward(
        self,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        camera_ids: Optional[torch.Tensor] = None,
        return_per_camera: bool = False
    ) -> torch.Tensor:
        """
        Encode camera parameters.

        Args:
            intrinsics: Camera intrinsic matrices [B, N, 3, 3]
            extrinsics: Camera extrinsic matrices [B, N, 3, 4]
            camera_ids: Camera IDs [B, N] (optional, defaults to 0...N-1)
            return_per_camera: If True, return per-camera embeddings [B, N, D]

        Returns:
            Camera embedding [B, output_dim] or [B, N, output_dim]
        """
        B, N = intrinsics.shape[:2]
        device = intrinsics.device

        # Normalize intrinsics
        intrinsics_norm = normalize_intrinsics(intrinsics, self.image_size)

        # Extract features
        intr_features = self._extract_intrinsics_features(intrinsics_norm)  # [B, N, 5]
        extr_features = self._extract_extrinsics_features(extrinsics)  # [B, N, 9]

        # Encode
        intr_embed = self.intrinsics_encoder(intr_features)  # [B, N, embed_dim]
        extr_embed = self.extrinsics_encoder(extr_features)  # [B, N, embed_dim]

        # Camera ID embedding
        if camera_ids is None:
            camera_ids = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        camera_embed = self.camera_id_embed(camera_ids)  # [B, N, embed_dim]

        # Fuse per-camera features
        fused = torch.cat([intr_embed, extr_embed, camera_embed], dim=-1)  # [B, N, embed_dim*3]
        per_camera_embed = self.fusion(fused)  # [B, N, output_dim]

        if return_per_camera:
            return per_camera_embed

        # Aggregate across cameras
        flat = per_camera_embed.reshape(B, -1)  # [B, N * output_dim]

        # Pad if fewer cameras than max
        if N < self.num_cameras:
            padding = torch.zeros(B, (self.num_cameras - N) * self.output_dim, device=device)
            flat = torch.cat([flat, padding], dim=-1)

        aggregated = self.aggregation(flat)  # [B, output_dim]

        return aggregated


# =============================================================================
# Road Topology Encoder
# =============================================================================

class RoadTopologyEncoder(nn.Module):
    """
    Encode road topology information including:
    - Road graph as sequence of waypoints
    - Road type (highway, urban, intersection)
    - Lane information (number of lanes, current lane)
    - Upcoming turns/intersections

    Uses cross-attention to condition transformer on road context.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        output_dim: int = 256,
        max_waypoints: int = 50,
        max_lanes: int = 8,
        num_road_types: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.max_waypoints = max_waypoints
        self.max_lanes = max_lanes

        # Waypoint encoder (x, y, z, heading, curvature, speed_limit)
        self.waypoint_encoder = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Positional encoding for waypoints (temporal ordering)
        self.waypoint_pos_embed = nn.Parameter(torch.randn(1, max_waypoints, embed_dim) * 0.02)

        # Road type embedding
        self.road_type_embed = nn.Embedding(num_road_types, embed_dim)

        # Lane information encoder
        self.lane_encoder = nn.Sequential(
            nn.Linear(3, embed_dim // 2),  # num_lanes, current_lane, lane_width
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Turn/intersection encoder
        self.turn_encoder = nn.Sequential(
            nn.Linear(5, embed_dim),  # turn_angle, distance_to_turn, turn_type (one-hot 3)
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Self-attention for waypoints
        self.waypoint_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention for conditioning (road context -> model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Key/value projection for cross-attention
        self.kv_proj = nn.Linear(embed_dim, output_dim)

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, output_dim),
            nn.Dropout(dropout)
        )

        # Layer norms
        self.norm_waypoints = nn.LayerNorm(embed_dim)
        self.norm_cross = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        waypoints: torch.Tensor,
        road_type: torch.Tensor,
        lane_info: torch.Tensor,
        turn_info: Optional[torch.Tensor] = None,
        waypoint_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode road topology.

        Args:
            waypoints: Waypoint features [B, N_waypoints, 6] (x, y, z, heading, curvature, speed_limit)
            road_type: Road type IDs [B] (enum index)
            lane_info: Lane features [B, 3] (num_lanes, current_lane, lane_width)
            turn_info: Turn features [B, 5] (turn_angle, distance, turn_type one-hot) (optional)
            waypoint_mask: Mask for valid waypoints [B, N_waypoints] (optional)

        Returns:
            Tuple of:
                - Global road embedding [B, output_dim]
                - Cross-attention context [B, N_waypoints, output_dim] for transformer conditioning
        """
        B, N = waypoints.shape[:2]
        device = waypoints.device

        # Encode waypoints
        waypoint_embed = self.waypoint_encoder(waypoints)  # [B, N, embed_dim]

        # Add positional encoding
        pos_embed = self.waypoint_pos_embed[:, :N, :]  # [1, N, embed_dim]
        waypoint_embed = waypoint_embed + pos_embed

        # Self-attention over waypoints
        waypoint_attn, _ = self.waypoint_attn(
            waypoint_embed, waypoint_embed, waypoint_embed,
            key_padding_mask=~waypoint_mask if waypoint_mask is not None else None
        )
        waypoint_embed = self.norm_waypoints(waypoint_embed + waypoint_attn)

        # Encode road type
        road_type_embed = self.road_type_embed(road_type)  # [B, embed_dim]

        # Encode lane info
        lane_embed = self.lane_encoder(lane_info)  # [B, embed_dim]

        # Encode turn info
        if turn_info is not None:
            turn_embed = self.turn_encoder(turn_info)  # [B, embed_dim]
        else:
            turn_embed = torch.zeros(B, self.embed_dim, device=device)

        # Global waypoint summary (mean pooling with mask)
        if waypoint_mask is not None:
            mask_expanded = waypoint_mask.unsqueeze(-1).float()
            waypoint_sum = (waypoint_embed * mask_expanded).sum(dim=1)
            waypoint_mean = waypoint_sum / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            waypoint_mean = waypoint_embed.mean(dim=1)

        # Fuse all features for global embedding
        fused = torch.cat([waypoint_mean, road_type_embed, lane_embed, turn_embed], dim=-1)
        global_embed = self.fusion(fused)  # [B, output_dim]

        # Create cross-attention context
        cross_attn_context = self.kv_proj(waypoint_embed)  # [B, N, output_dim]

        return global_embed, cross_attn_context

    def apply_cross_attention(
        self,
        query: torch.Tensor,
        road_context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply cross-attention from model queries to road context.

        Args:
            query: Model hidden states [B, T, output_dim]
            road_context: Road cross-attention context [B, N_waypoints, output_dim]
            key_padding_mask: Mask for road context [B, N_waypoints]

        Returns:
            Cross-attended features [B, T, output_dim]
        """
        attn_out, _ = self.cross_attn(
            query, road_context, road_context,
            key_padding_mask=~key_padding_mask if key_padding_mask is not None else None
        )
        return self.norm_cross(query + attn_out)


# =============================================================================
# 3D Bounding Box Encoder
# =============================================================================

class BoundingBox3DEncoder(nn.Module):
    """
    Encode detected objects with 3D bounding boxes.

    Features per object:
    - 3D center position (x, y, z)
    - 3D dimensions (length, width, height)
    - Heading/yaw angle
    - Velocity (vx, vy, vz)
    - Object class embedding
    - Detection confidence

    Supports distance-based attention weighting for importance.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        output_dim: int = 256,
        max_objects: int = 64,
        num_object_classes: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.max_objects = max_objects
        self.num_object_classes = num_object_classes

        # Geometry encoder (position + dimensions + heading = 7)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(7, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Velocity encoder
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # Object class embedding
        self.class_embed = nn.Embedding(num_object_classes, embed_dim)

        # Confidence encoding
        self.confidence_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.SiLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )

        # Distance-based positional encoding
        self.distance_proj = nn.Linear(embed_dim, 1)  # For attention weighting

        # Self-attention over objects
        self.object_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, output_dim),
            nn.Dropout(dropout)
        )

        # Output projection for memory tokens
        self.memory_proj = nn.Linear(embed_dim, output_dim)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _compute_distance_weights(
        self,
        positions: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute distance-based attention weights.

        Args:
            positions: Object positions [B, N, 3]
            object_mask: Valid object mask [B, N]

        Returns:
            Attention weights [B, N]
        """
        # Compute Euclidean distance from ego (assumed at origin)
        distances = torch.norm(positions, dim=-1)  # [B, N]

        # Inverse distance weighting (closer objects more important)
        # Add small epsilon to avoid division by zero
        weights = 1.0 / (distances + 1.0)  # 1/(d+1) gives [0, 1] range

        # Apply mask
        if object_mask is not None:
            weights = weights * object_mask.float()

        # Normalize
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return weights

    def forward(
        self,
        positions: torch.Tensor,
        dimensions: torch.Tensor,
        headings: torch.Tensor,
        velocities: torch.Tensor,
        object_classes: torch.Tensor,
        confidences: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode 3D bounding boxes.

        Args:
            positions: Object centers [B, N, 3] (x, y, z in ego frame)
            dimensions: Object sizes [B, N, 3] (length, width, height)
            headings: Object yaw angles [B, N] (radians)
            velocities: Object velocities [B, N, 3] (vx, vy, vz)
            object_classes: Object class IDs [B, N]
            confidences: Detection confidences [B, N]
            object_mask: Valid object mask [B, N] (optional)

        Returns:
            Tuple of:
                - Global object embedding [B, output_dim]
                - Memory tokens for cross-attention [B, N, output_dim]
        """
        B, N = positions.shape[:2]
        device = positions.device

        # Normalize positions and dimensions
        positions_norm = positions / 100.0  # Assume max 100m range
        dimensions_norm = dimensions / 10.0  # Assume max 10m dimension
        headings_norm = headings / math.pi  # [-1, 1] range
        velocities_norm = velocities / 30.0  # Assume max 30 m/s

        # Create geometry features
        geometry = torch.cat([
            positions_norm,
            dimensions_norm,
            headings_norm.unsqueeze(-1)
        ], dim=-1)  # [B, N, 7]

        # Encode features
        geom_embed = self.geometry_encoder(geometry)  # [B, N, embed_dim]
        vel_embed = self.velocity_encoder(velocities_norm)  # [B, N, embed_dim]
        class_embed = self.class_embed(object_classes)  # [B, N, embed_dim]
        conf_embed = self.confidence_encoder(confidences.unsqueeze(-1))  # [B, N, embed_dim]

        # Fuse per-object features
        fused = torch.cat([geom_embed, vel_embed, class_embed, conf_embed], dim=-1)
        object_embed = self.fusion(fused)  # [B, N, output_dim]

        # Convert back to embed_dim for attention
        object_features = self.norm(geom_embed + vel_embed + class_embed + conf_embed)

        # Self-attention over objects
        key_padding_mask = ~object_mask if object_mask is not None else None
        object_attn, _ = self.object_attn(
            object_features, object_features, object_features,
            key_padding_mask=key_padding_mask
        )
        object_features = self.norm(object_features + object_attn)

        # Compute distance-based weights
        distance_weights = self._compute_distance_weights(positions, object_mask)  # [B, N]

        # Weighted global embedding
        weighted_features = object_features * distance_weights.unsqueeze(-1)
        global_embed = weighted_features.sum(dim=1)  # [B, embed_dim]

        # Project to output dimension
        global_embed = self.memory_proj(global_embed)  # [B, output_dim]

        # Memory tokens for cross-attention
        memory_tokens = self.memory_proj(object_features)  # [B, N, output_dim]

        return global_embed, memory_tokens


# =============================================================================
# Ego State Encoder
# =============================================================================

class EgoStateEncoder(nn.Module):
    """
    Encode ego vehicle state:
    - Current position (x, y, z in global frame)
    - Velocity (vx, vy, vz)
    - Acceleration (ax, ay, az)
    - Heading and heading rate
    - GPS coordinates (normalized)
    - IMU data (optional)
    """

    def __init__(
        self,
        embed_dim: int = 64,
        output_dim: int = 256,
        use_gps: bool = True,
        use_imu: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.use_gps = use_gps
        self.use_imu = use_imu

        # Position encoder (3D)
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Velocity encoder (3D)
        self.velocity_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Acceleration encoder (3D)
        self.acceleration_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Heading encoder (heading + heading_rate)
        self.heading_encoder = nn.Sequential(
            nn.Linear(2, embed_dim // 2),
            nn.SiLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        # GPS encoder (lat, lon normalized)
        if use_gps:
            self.gps_encoder = nn.Sequential(
                nn.Linear(2, embed_dim // 2),
                nn.SiLU(),
                nn.Linear(embed_dim // 2, embed_dim)
            )

        # IMU encoder (angular velocity + linear acceleration = 6 values)
        if use_imu:
            self.imu_encoder = nn.Sequential(
                nn.Linear(6, embed_dim),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim)
            )

        # Fusion network
        num_inputs = 4  # position, velocity, acceleration, heading
        if use_gps:
            num_inputs += 1
        if use_imu:
            num_inputs += 1

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * num_inputs, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, output_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        heading: torch.Tensor,
        heading_rate: torch.Tensor,
        gps: Optional[torch.Tensor] = None,
        imu: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode ego state.

        Args:
            position: Ego position [B, 3] (x, y, z)
            velocity: Ego velocity [B, 3] (vx, vy, vz)
            acceleration: Ego acceleration [B, 3] (ax, ay, az)
            heading: Heading angle [B] (radians)
            heading_rate: Heading rate [B] (rad/s)
            gps: GPS coordinates [B, 2] (lat, lon) normalized to [0, 1]
            imu: IMU data [B, 6] (angular_vel, linear_accel)

        Returns:
            Ego state embedding [B, output_dim]
        """
        B = position.shape[0]
        device = position.device

        # Normalize inputs
        position_norm = position / 1000.0  # Assume max 1km from origin
        velocity_norm = velocity / 50.0  # Assume max 50 m/s
        acceleration_norm = acceleration / 10.0  # Assume max 10 m/s^2
        heading_norm = torch.stack([
            torch.sin(heading),
            torch.cos(heading)
        ], dim=-1) / math.pi
        heading_rate_norm = heading_rate / (2 * math.pi)  # Normalize

        # Encode components
        pos_embed = self.position_encoder(position_norm)
        vel_embed = self.velocity_encoder(velocity_norm)
        accel_embed = self.acceleration_encoder(acceleration_norm)

        heading_features = torch.cat([
            heading_norm,
            heading_rate_norm.unsqueeze(-1)
        ], dim=-1)[:, :2]  # Take first 2 features
        heading_embed = self.heading_encoder(heading_features)

        # Collect all embeddings
        embeddings = [pos_embed, vel_embed, accel_embed, heading_embed]

        # Optional GPS
        if self.use_gps and gps is not None:
            gps_embed = self.gps_encoder(gps)
            embeddings.append(gps_embed)
        elif self.use_gps:
            embeddings.append(torch.zeros(B, self.embed_dim, device=device))

        # Optional IMU
        if self.use_imu and imu is not None:
            imu_norm = imu.clone()
            imu_norm[:, :3] /= 10.0  # Angular velocity
            imu_norm[:, 3:] /= 50.0  # Linear acceleration
            imu_embed = self.imu_encoder(imu_norm)
            embeddings.append(imu_embed)
        elif self.use_imu:
            embeddings.append(torch.zeros(B, self.embed_dim, device=device))

        # Fuse all
        fused = torch.cat(embeddings, dim=-1)
        ego_embed = self.fusion(fused)

        return ego_embed


# =============================================================================
# Temporal/Environmental Conditioning
# =============================================================================

class TemporalEnvironmentalEncoder(nn.Module):
    """
    Encode temporal and environmental conditions:
    - Time of day (continuous + cyclic encoding)
    - Weather conditions (categorical)
    - Season/lighting conditions
    - Sun position (azimuth, elevation)
    """

    def __init__(
        self,
        time_embed_dim: int = 64,
        weather_embed_dim: int = 32,
        output_dim: int = 256,
        num_weather_conditions: int = 8,
        num_seasons: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.weather_embed_dim = weather_embed_dim
        self.output_dim = output_dim

        # Time of day encoder (hour as cyclic features)
        self.time_encoder = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # Weather condition embedding
        self.weather_embed = nn.Embedding(num_weather_conditions, weather_embed_dim)

        # Weather intensity (0-1)
        self.weather_intensity_encoder = nn.Sequential(
            nn.Linear(1, weather_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(weather_embed_dim // 2, weather_embed_dim)
        )

        # Season embedding
        self.season_embed = nn.Embedding(num_seasons, weather_embed_dim)

        # Sun position encoder (azimuth, elevation)
        self.sun_encoder = nn.Sequential(
            nn.Linear(4, weather_embed_dim),  # sin/cos for azimuth, sin/cos for elevation
            nn.SiLU(),
            nn.Linear(weather_embed_dim, weather_embed_dim)
        )

        # Lighting conditions (ambient + direct)
        self.lighting_encoder = nn.Sequential(
            nn.Linear(2, weather_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(weather_embed_dim // 2, weather_embed_dim)
        )

        # Fusion
        total_dim = time_embed_dim + weather_embed_dim * 5  # time + weather + intensity + season + sun + lighting
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        time_of_day: torch.Tensor,
        weather: torch.Tensor,
        weather_intensity: Optional[torch.Tensor] = None,
        season: Optional[torch.Tensor] = None,
        sun_azimuth: Optional[torch.Tensor] = None,
        sun_elevation: Optional[torch.Tensor] = None,
        lighting: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode temporal and environmental conditions.

        Args:
            time_of_day: Hour of day [B] (0-24)
            weather: Weather condition ID [B]
            weather_intensity: Weather intensity [B] (0-1)
            season: Season ID [B] (0-3)
            sun_azimuth: Sun azimuth angle [B] (radians)
            sun_elevation: Sun elevation angle [B] (radians)
            lighting: Lighting conditions [B, 2] (ambient, direct)

        Returns:
            Temporal/environmental embedding [B, output_dim]
        """
        B = time_of_day.shape[0]
        device = time_of_day.device

        # Create sinusoidal time embedding
        time_embed = sinusoidal_embedding(time_of_day / 24.0, self.time_embed_dim)
        time_embed = self.time_encoder(time_embed)

        # Weather embedding
        weather_embed = self.weather_embed(weather)

        # Weather intensity
        if weather_intensity is not None:
            intensity_embed = self.weather_intensity_encoder(weather_intensity.unsqueeze(-1))
        else:
            intensity_embed = torch.zeros(B, self.weather_embed_dim, device=device)

        # Season embedding
        if season is not None:
            season_embed = self.season_embed(season)
        else:
            season_embed = torch.zeros(B, self.weather_embed_dim, device=device)

        # Sun position encoding
        if sun_azimuth is not None and sun_elevation is not None:
            sun_features = torch.stack([
                torch.sin(sun_azimuth),
                torch.cos(sun_azimuth),
                torch.sin(sun_elevation),
                torch.cos(sun_elevation)
            ], dim=-1)
            sun_embed = self.sun_encoder(sun_features)
        else:
            sun_embed = torch.zeros(B, self.weather_embed_dim, device=device)

        # Lighting conditions
        if lighting is not None:
            lighting_embed = self.lighting_encoder(lighting)
        else:
            lighting_embed = torch.zeros(B, self.weather_embed_dim, device=device)

        # Fuse all
        fused = torch.cat([
            time_embed,
            weather_embed,
            intensity_embed,
            season_embed,
            sun_embed,
            lighting_embed
        ], dim=-1)

        return self.fusion(fused)


# =============================================================================
# Integration Module (AdaLN + CFG Support)
# =============================================================================

class AdaLNModulation(nn.Module):
    """
    Adaptive Layer Normalization modulation for DiT-style conditioning.
    Produces scale, shift, and gate parameters for attention and MLP sublayers.
    """

    def __init__(
        self,
        cond_dim: int,
        model_dim: int,
        num_outputs: int = 6  # scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp
    ):
        super().__init__()
        self.model_dim = model_dim
        self.num_outputs = num_outputs

        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_outputs * model_dim)
        )

        # Initialize to identity
        with torch.no_grad():
            self.projection[1].weight.zero_()
            self.projection[1].bias.zero_()
            # Set scales to 1 and shifts/gates to 0
            self.projection[1].bias[0:model_dim] = 1.0  # scale_attn
            self.projection[1].bias[3*model_dim:4*model_dim] = 1.0  # scale_mlp

    def forward(self, conditioning: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Compute modulation parameters.

        Args:
            conditioning: Conditioning vector [B, cond_dim]

        Returns:
            Tuple of 6 tensors each [B, model_dim]:
            (scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp)
        """
        params = self.projection(conditioning)  # [B, num_outputs * model_dim]
        params = params.chunk(self.num_outputs, dim=-1)
        return params


class ConditioningIntegration(nn.Module):
    """
    Integration module combining all conditioning signals.

    Features:
    - Combines camera, road, objects, ego, and temporal conditioning
    - Outputs AdaLN modulation parameters per layer
    - Supports classifier-free guidance with conditioning dropout
    - Provides cross-attention contexts for detailed conditioning
    """

    def __init__(
        self,
        config: ConditioningConfig
    ):
        super().__init__()
        self.config = config

        # Individual encoders
        self.camera_encoder = CameraEncoder(
            embed_dim=config.camera_embed_dim,
            output_dim=config.cond_dim,
            num_cameras=config.num_cameras,
            dropout=config.dropout
        )

        self.road_encoder = RoadTopologyEncoder(
            embed_dim=config.road_embed_dim,
            output_dim=config.cond_dim,
            max_waypoints=config.max_waypoints,
            max_lanes=config.max_lanes,
            num_heads=config.road_cross_attn_heads,
            dropout=config.dropout
        )

        self.bbox_encoder = BoundingBox3DEncoder(
            embed_dim=config.bbox_embed_dim,
            output_dim=config.cond_dim,
            max_objects=config.max_objects,
            num_object_classes=config.num_object_classes,
            dropout=config.dropout
        )

        self.ego_encoder = EgoStateEncoder(
            embed_dim=config.ego_state_dim,
            output_dim=config.cond_dim,
            dropout=config.dropout
        )

        self.temporal_encoder = TemporalEnvironmentalEncoder(
            time_embed_dim=config.time_embed_dim,
            weather_embed_dim=config.weather_embed_dim,
            output_dim=config.cond_dim,
            dropout=config.dropout
        )

        # Conditioning fusion
        self.conditioning_fusion = nn.Sequential(
            nn.Linear(config.cond_dim * 5, config.cond_dim * 2),
            nn.SiLU(),
            nn.Linear(config.cond_dim * 2, config.cond_dim),
            nn.Dropout(config.dropout)
        )

        # AdaLN modulation producer
        self.adaln_modulation = AdaLNModulation(
            cond_dim=config.cond_dim,
            model_dim=config.model_dim,
            num_outputs=config.num_adaln_modulation_outputs
        )

        # Null conditioning for CFG (learnable)
        self.null_conditioning = nn.Parameter(torch.zeros(config.cond_dim))

        # Layer norm for conditioning
        self.cond_norm = nn.LayerNorm(config.cond_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.null_conditioning, std=0.02)

    def _apply_cfg_dropout(
        self,
        conditioning: torch.Tensor,
        training: bool,
        drop_prob: float
    ) -> torch.Tensor:
        """
        Apply conditioning dropout for classifier-free guidance.

        Args:
            conditioning: Conditioning vector [B, cond_dim]
            training: Whether in training mode
            drop_prob: Dropout probability

        Returns:
            Conditioning with dropout applied [B, cond_dim]
        """
        if not training or drop_prob == 0.0:
            return conditioning

        B = conditioning.shape[0]
        mask = torch.rand(B, 1, device=conditioning.device) > drop_prob

        # Replace dropped samples with null conditioning
        null_cond = self.null_conditioning.unsqueeze(0).expand(B, -1)
        conditioning = torch.where(mask, conditioning, null_cond)

        return conditioning

    def forward(
        self,
        camera_data: Optional[Dict[str, torch.Tensor]] = None,
        road_data: Optional[Dict[str, torch.Tensor]] = None,
        bbox_data: Optional[Dict[str, torch.Tensor]] = None,
        ego_data: Optional[Dict[str, torch.Tensor]] = None,
        temporal_data: Optional[Dict[str, torch.Tensor]] = None,
        cfg_scale: float = 1.0,
        return_cross_attention_contexts: bool = True
    ) -> Dict[str, Any]:
        """
        Compute integrated conditioning.

        Args:
            camera_data: Dict with 'intrinsics', 'extrinsics' tensors
            road_data: Dict with 'waypoints', 'road_type', 'lane_info', etc.
            bbox_data: Dict with 'positions', 'dimensions', 'headings', etc.
            ego_data: Dict with 'position', 'velocity', 'acceleration', etc.
            temporal_data: Dict with 'time_of_day', 'weather', etc.
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            return_cross_attention_contexts: Whether to return contexts for cross-attention

        Returns:
            Dict containing:
                - 'conditioning': Global conditioning vector [B, cond_dim]
                - 'adaln_params': Tuple of AdaLN modulation parameters
                - 'road_context': Road cross-attention context (if requested)
                - 'object_tokens': Object memory tokens (if requested)
        """
        B = self._get_batch_size(camera_data, road_data, bbox_data, ego_data, temporal_data)
        device = self._get_device(camera_data, road_data, bbox_data, ego_data, temporal_data)

        # Encode each modality
        conditioning_components = []
        cross_attention_contexts = {}

        # Camera conditioning
        if camera_data is not None:
            camera_cond = self.camera_encoder(
                intrinsics=camera_data['intrinsics'],
                extrinsics=camera_data['extrinsics'],
                camera_ids=camera_data.get('camera_ids')
            )
        else:
            camera_cond = torch.zeros(B, self.config.cond_dim, device=device)
        conditioning_components.append(camera_cond)

        # Road conditioning
        if road_data is not None:
            road_cond, road_context = self.road_encoder(
                waypoints=road_data['waypoints'],
                road_type=road_data['road_type'],
                lane_info=road_data['lane_info'],
                turn_info=road_data.get('turn_info'),
                waypoint_mask=road_data.get('waypoint_mask')
            )
            if return_cross_attention_contexts:
                cross_attention_contexts['road_context'] = road_context
                cross_attention_contexts['road_mask'] = road_data.get('waypoint_mask')
        else:
            road_cond = torch.zeros(B, self.config.cond_dim, device=device)
        conditioning_components.append(road_cond)

        # 3D bounding box conditioning
        if bbox_data is not None:
            bbox_cond, object_tokens = self.bbox_encoder(
                positions=bbox_data['positions'],
                dimensions=bbox_data['dimensions'],
                headings=bbox_data['headings'],
                velocities=bbox_data['velocities'],
                object_classes=bbox_data['object_classes'],
                confidences=bbox_data['confidences'],
                object_mask=bbox_data.get('object_mask')
            )
            if return_cross_attention_contexts:
                cross_attention_contexts['object_tokens'] = object_tokens
                cross_attention_contexts['object_mask'] = bbox_data.get('object_mask')
        else:
            bbox_cond = torch.zeros(B, self.config.cond_dim, device=device)
        conditioning_components.append(bbox_cond)

        # Ego state conditioning
        if ego_data is not None:
            ego_cond = self.ego_encoder(
                position=ego_data['position'],
                velocity=ego_data['velocity'],
                acceleration=ego_data['acceleration'],
                heading=ego_data['heading'],
                heading_rate=ego_data['heading_rate'],
                gps=ego_data.get('gps'),
                imu=ego_data.get('imu')
            )
        else:
            ego_cond = torch.zeros(B, self.config.cond_dim, device=device)
        conditioning_components.append(ego_cond)

        # Temporal/environmental conditioning
        if temporal_data is not None:
            temporal_cond = self.temporal_encoder(
                time_of_day=temporal_data['time_of_day'],
                weather=temporal_data['weather'],
                weather_intensity=temporal_data.get('weather_intensity'),
                season=temporal_data.get('season'),
                sun_azimuth=temporal_data.get('sun_azimuth'),
                sun_elevation=temporal_data.get('sun_elevation'),
                lighting=temporal_data.get('lighting')
            )
        else:
            temporal_cond = torch.zeros(B, self.config.cond_dim, device=device)
        conditioning_components.append(temporal_cond)

        # Fuse all conditioning
        combined = torch.cat(conditioning_components, dim=-1)  # [B, cond_dim * 5]
        conditioning = self.conditioning_fusion(combined)  # [B, cond_dim]
        conditioning = self.cond_norm(conditioning)

        # Apply CFG dropout during training
        conditioning = self._apply_cfg_dropout(
            conditioning,
            self.training,
            self.config.cfg_dropout_prob
        )

        # Compute AdaLN modulation parameters
        adaln_params = self.adaln_modulation(conditioning)

        # Build output
        output = {
            'conditioning': conditioning,
            'adaln_params': adaln_params,
        }

        if return_cross_attention_contexts:
            output.update(cross_attention_contexts)

        return output

    def _get_batch_size(self, *data_dicts) -> int:
        """Get batch size from first available tensor."""
        for data in data_dicts:
            if data is not None:
                for v in data.values():
                    if isinstance(v, torch.Tensor):
                        return v.shape[0]
        return 1

    def _get_device(self, *data_dicts) -> torch.device:
        """Get device from first available tensor."""
        for data in data_dicts:
            if data is not None:
                for v in data.values():
                    if isinstance(v, torch.Tensor):
                        return v.device
        return torch.device('cpu')

    def get_null_conditioning(self, batch_size: int, device: torch.device) -> Dict[str, Any]:
        """
        Get null conditioning for classifier-free guidance.

        Args:
            batch_size: Batch size
            device: Device to place tensors on

        Returns:
            Dict with null conditioning outputs
        """
        null_cond = self.null_conditioning.unsqueeze(0).expand(batch_size, -1)
        null_cond = null_cond.to(device)

        adaln_params = self.adaln_modulation(null_cond)

        return {
            'conditioning': null_cond,
            'adaln_params': adaln_params,
        }

    def apply_cfg(
        self,
        conditional_output: torch.Tensor,
        unconditional_output: torch.Tensor,
        cfg_scale: float
    ) -> torch.Tensor:
        """
        Apply classifier-free guidance.

        Args:
            conditional_output: Model output with conditioning
            unconditional_output: Model output without conditioning
            cfg_scale: Guidance scale (1.0 = no guidance)

        Returns:
            Guided output
        """
        return unconditional_output + cfg_scale * (conditional_output - unconditional_output)


# =============================================================================
# Utility Functions for Integration with World Model
# =============================================================================

def create_conditioning_module(
    model_dim: int = 512,
    cond_dim: int = 256,
    num_cameras: int = 6,
    max_waypoints: int = 50,
    max_objects: int = 64,
    cfg_dropout_prob: float = 0.1,
    dropout: float = 0.1
) -> ConditioningIntegration:
    """
    Factory function to create conditioning module.

    Args:
        model_dim: Model dimension for AdaLN outputs
        cond_dim: Conditioning embedding dimension
        num_cameras: Number of cameras in multi-camera setup
        max_waypoints: Maximum number of road waypoints
        max_objects: Maximum number of detected objects
        cfg_dropout_prob: Dropout probability for CFG
        dropout: General dropout probability

    Returns:
        Configured ConditioningIntegration module
    """
    config = ConditioningConfig(
        model_dim=model_dim,
        cond_dim=cond_dim,
        num_cameras=num_cameras,
        max_waypoints=max_waypoints,
        max_objects=max_objects,
        cfg_dropout_prob=cfg_dropout_prob,
        dropout=dropout
    )

    return ConditioningIntegration(config)


# =============================================================================
# Testing
# =============================================================================

def test_conditioning_module():
    """Test the conditioning module with synthetic data."""
    print("Testing GAIA-2 style conditioning module...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 4  # Batch size

    # Create module
    config = ConditioningConfig(
        model_dim=512,
        cond_dim=256,
        num_cameras=6,
        max_waypoints=50,
        max_objects=32
    )
    module = ConditioningIntegration(config).to(device)

    # Create synthetic data
    camera_data = {
        'intrinsics': torch.randn(B, 6, 3, 3, device=device),
        'extrinsics': torch.randn(B, 6, 3, 4, device=device),
    }

    road_data = {
        'waypoints': torch.randn(B, 50, 6, device=device),
        'road_type': torch.randint(0, 8, (B,), device=device),
        'lane_info': torch.randn(B, 3, device=device),
        'waypoint_mask': torch.ones(B, 50, dtype=torch.bool, device=device),
    }

    bbox_data = {
        'positions': torch.randn(B, 32, 3, device=device) * 50,
        'dimensions': torch.abs(torch.randn(B, 32, 3, device=device)) + 1,
        'headings': torch.randn(B, 32, device=device) * math.pi,
        'velocities': torch.randn(B, 32, 3, device=device) * 10,
        'object_classes': torch.randint(0, 11, (B, 32), device=device),
        'confidences': torch.rand(B, 32, device=device),
        'object_mask': torch.ones(B, 32, dtype=torch.bool, device=device),
    }

    ego_data = {
        'position': torch.randn(B, 3, device=device) * 100,
        'velocity': torch.randn(B, 3, device=device) * 10,
        'acceleration': torch.randn(B, 3, device=device),
        'heading': torch.randn(B, device=device) * math.pi,
        'heading_rate': torch.randn(B, device=device) * 0.5,
    }

    temporal_data = {
        'time_of_day': torch.rand(B, device=device) * 24,
        'weather': torch.randint(0, 8, (B,), device=device),
    }

    # Forward pass
    outputs = module(
        camera_data=camera_data,
        road_data=road_data,
        bbox_data=bbox_data,
        ego_data=ego_data,
        temporal_data=temporal_data,
    )

    print(f"Conditioning shape: {outputs['conditioning'].shape}")
    print(f"Number of AdaLN params: {len(outputs['adaln_params'])}")
    print(f"AdaLN param shapes: {[p.shape for p in outputs['adaln_params']]}")

    if 'road_context' in outputs:
        print(f"Road context shape: {outputs['road_context'].shape}")
    if 'object_tokens' in outputs:
        print(f"Object tokens shape: {outputs['object_tokens'].shape}")

    # Test null conditioning
    null_outputs = module.get_null_conditioning(B, device)
    print(f"Null conditioning shape: {null_outputs['conditioning'].shape}")

    # Test with partial conditioning
    outputs_partial = module(
        camera_data=camera_data,
        ego_data=ego_data,
    )
    print(f"Partial conditioning shape: {outputs_partial['conditioning'].shape}")

    # Count parameters
    num_params = sum(p.numel() for p in module.parameters())
    print(f"Total parameters: {num_params:,}")

    print("Conditioning module test completed successfully!")


if __name__ == "__main__":
    test_conditioning_module()
