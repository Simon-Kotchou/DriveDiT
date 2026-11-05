"""The first dc-2.0 production session (T0L, Arland, 2026-11-05), read-only: the shard has one decision step per
decision callback (3101), and its action_cmd and state statistics, masks and segments equal the reader's.

Skipped when the session is not on this machine (CI). Reading it decodes 3101 fiducials from 5.9 GB of PNGs, about
8 minutes on /mnt/e, so the session is read once per module. The shard goes to the test scratch, never into the
captures tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from drivedit.data import build
from drivedit.data.shard import Shard, WindowIndex
from drivedit.data.spec import FeatureSpec

REAL = Path("/mnt/e/DriveDiT/captures/arland/2026-11-05/T0L/arland_T0L_20261105T072648Z_wb_i00_66b8")

pytestmark = pytest.mark.skipif(not (REAL / "session.json").is_file(), reason="the T0L session is not on this machine")


@pytest.fixture(scope="module")
def real_session() -> Any:
    return build.open_dc2_session(REAL)


@pytest.fixture(scope="module")
def real_shard(real_session: Any, spec: FeatureSpec, scratch: Path) -> Shard:
    return Shard(build.write_shard(real_session, scratch / "real_root", spec=spec))


def _reader_rows(session: Any, key: str, width: int) -> np.ndarray:
    """[T, width] of the reader's decision samples, NaN for None (a missing row or the no-command sentinel)."""
    rows = [
        [np.nan] * width if d[key] is None else [np.nan if v is None else float(v) for v in d[key]]
        for d in session.decisions
    ]
    return np.array(rows, dtype=np.float64)


def test_one_step_per_decision_callback(real_shard: Shard, real_session: Any) -> None:
    # 620.1 s of world time on the 200 ms grid from the first ctl row: grid points 0..3100, none skipped
    assert real_shard.n_steps == len(real_session.decisions) == 3101
    assert real_session.decision_grid_skipped == 0
    assert [int(d["grid_point"]) for d in real_session.decisions] == list(range(3101))
    assert int((real_shard.array("slot_id") >= 0).sum()) == 3100  # obs_t valid at every step but the first
    assert real_shard.meta["missing"].keys() == {"road_context"}
    for slot in real_shard.array("slot_id")[::500]:
        if slot >= 0:
            assert real_shard.image_path(int(slot)).is_file()


def test_action_cmd_and_state_statistics_match_the_reader(real_shard: Shard, real_session: Any) -> None:
    for name, width in (("action_cmd", 3), ("state", 4)):
        want = _reader_rows(real_session, name, width)
        got = np.asarray(real_shard.array(name), dtype=np.float64)
        assert got.shape == want.shape == (3101, width), name
        np.testing.assert_array_equal(np.isnan(got), np.isnan(want), err_msg=name)
        for stat in (np.nanmean, np.nanstd, np.nanmin, np.nanmax):
            np.testing.assert_allclose(stat(got, axis=0), stat(want, axis=0), rtol=1e-5, atol=1e-6, err_msg=name)
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, equal_nan=True, err_msg=name)
    # mode E without noise: the applied command is the expert label wherever both exist
    cmd, label = real_shard.array("action_cmd"), real_shard.array("action_label")
    both = ~np.isnan(cmd).any(axis=1) & ~np.isnan(label).any(axis=1)
    assert int(both.sum()) > 2900
    np.testing.assert_array_equal(cmd[both], label[both])


def test_masks_and_segments_match_the_reader(real_shard: Shard, real_session: Any) -> None:
    names = real_shard.meta["channels"]["masks"]
    want = np.array([[d["masks"][n] for n in names] for d in real_session.decisions], dtype=np.uint8)
    np.testing.assert_array_equal(real_shard.array("masks"), want)  # nav_valid 1 on the whole loop
    assert real_shard.meta["counts"]["nav_invalid_in_route_episodes"] == 0
    steps = real_shard.array("step")
    got = [(int(steps[s.start]), int(steps[s.stop - 1]), s.length) for s in real_shard.segments]
    assert got == [(s["first_decision_step"], s["last_decision_step"], s["n_steps"]) for s in real_session.segments]
    assert [s.length for s in real_shard.segments] == [1275, 1275, 416]
    index = WindowIndex([real_shard], 10)
    assert len(index) == sum(s.length - 9 for s in real_shard.segments)


def test_contact_and_route_provenance(real_shard: Shard) -> None:
    contacts = real_shard.array("contact_events")
    (t,) = np.flatnonzero(contacts[:, 0] == 1).tolist()
    steps = real_shard.array("step")
    assert steps[t] < 11918 <= steps[t + 1]  # the src=contact terrain touch, onset_step 11918
    assert real_shard.meta["categories"]["contact_class"][int(contacts[t, 1])] == "terrain"
    route = real_shard.meta["source"]["routes"]["arland_t0_loop"]
    full = "77838c1346f2c7a6b18be78570d9f8cb560ee10a0e69718b28eb0f028cc0f66f"  # the deployed version (78e539b)
    assert route["sha256"] == full
    assert route["logged_route_sha256"] == [
        {
            "logged": "6370fc78f4266db37367ce7a3110a2e2b7515aeb90f695382092948d946f0c9d",
            "form": "prefix_8191",
            "verified_against": full,
        }
    ]
