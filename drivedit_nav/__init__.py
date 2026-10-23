"""drivedit_nav: the one navigation function of capture contract dc-2.0 and its route schema (DD-11-min).

nav_v1(route, pose) gives the ego-frame route points 20/40 m ahead, the next-maneuver one-hot with its distance and
the commanded-stop distance (docs/capture_contract/dc-2.0.md section 13). The harvester, the dataset builder and the
closed-loop bridge all call this package, so the three paths agree exactly (C22). Pure Python, no dependencies.

Routes follow route_schema.json (drivedit.route/1). load_route refuses Everon routes, and any path under heldout/ or a
SEALED directory, unless it is called on the sealed evaluation path with sealed_eval=True.

DRAFT with capture contract dc-2.0: nav_v1 and the route schema freeze with the contract; any later change of nav
behaviour or of the schema is nav_v2 (dc-2.0.md section 0).
"""

from .nav_v1 import (
    D_MAX_M,
    LOOKAHEAD_M,
    MAP_SPLITS,
    NAV_CHANNELS,
    NAV_VERSION,
    PROGRESS_WINDOW_M,
    ROUTE_DEVIATION_M,
    ROUTE_SCHEMA_FILE,
    ROUTE_SCHEMA_ID,
    Pose,
    Route,
    RouteError,
    SealedMapError,
    canonical_json,
    load_route,
    nav_sequence,
    nav_v1,
    nav_vector,
    pose_from_transform,
    route_from_dict,
    to_ego,
    to_legacy_6d,
    validate_route,
)

__all__ = [
    "D_MAX_M", "LOOKAHEAD_M", "MAP_SPLITS", "NAV_CHANNELS", "NAV_VERSION", "PROGRESS_WINDOW_M", "ROUTE_DEVIATION_M",
    "ROUTE_SCHEMA_FILE", "ROUTE_SCHEMA_ID", "Pose", "Route", "RouteError", "SealedMapError", "canonical_json",
    "load_route", "nav_sequence", "nav_v1", "nav_vector", "pose_from_transform", "route_from_dict", "to_ego",
    "to_legacy_6d", "validate_route",
]
