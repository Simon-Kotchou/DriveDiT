"""Stratification of evaluation rows by speed (numpy only).

The speed bins are the capture program's: {0-5, 5-20, 20-35, 35-50, >50} km/h (research delta-2026-11 §3.3, "Speed
coverage"). The controllability gates are stated per bin there and in §5.2 ("EAI > 0 in every speed bin"), so the
metrics report them per bin.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Bin edges in km/h; a row falls in [edge_i, edge_i+1), and the last bin is open.
SPEED_BIN_EDGES_KMH = (0.0, 5.0, 20.0, 35.0, 50.0)
SPEED_BIN_NAMES = ("0-5", "5-20", "20-35", "35-50", ">50")
_MPS_TO_KMH = 3.6


def speed_bin(speed_mps: npt.ArrayLike) -> npt.NDArray[np.int64]:
    """Index into ``SPEED_BIN_NAMES`` of each row's absolute speed (features_v1 ``state.speed_mps`` is signed)."""
    kmh = np.abs(np.asarray(speed_mps, dtype=np.float64)) * _MPS_TO_KMH
    if not np.all(np.isfinite(kmh)):
        raise ValueError("speed_mps has non-finite values")
    out: npt.NDArray[np.int64] = np.searchsorted(np.asarray(SPEED_BIN_EDGES_KMH), kmh, side="right") - 1
    return out
