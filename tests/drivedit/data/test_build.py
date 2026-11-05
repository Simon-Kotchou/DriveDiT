"""drivedit.data.build on the golden synthetic dc-2.0 session: shapes and dtypes per features_v1, the contract
offsets (against the raw ctl/telemetry rows and against golden.json, which make_synthetic.py computed from its own
simulation), masks and segments, future_pose and the anchors, contact_events, the image source, refusals, the 10 Hz
override and determinism."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from drivedit.data import build
from drivedit.data.shard import Shard, WindowIndex
from drivedit.data.spec import FeatureSpec

from .conftest import CARD, GOLDEN_DIR, ROUTES_DIR

TOL = 1e-4


def _shard(path: Path) -> Shard:
    return Shard(path)


def test_shapes_and_dtypes_follow_features_v1(golden_shard_dir: Path, spec: FeatureSpec, golden_session: Any) -> None:
    shard = _shard(golden_shard_dir)
    n = len(golden_session.decisions)
    assert shard.n_steps == n == 68
    assert shard.meta["future_pose_horizon"] == 30
    for name in build.DEFAULT_TENSORS:
        if name == "road_context":
            continue
        array = shard.array(name)
        assert array.shape == spec.shape(name, n, decision_hz=5, horizon=30), name
        assert array.dtype == (np.uint8 if name == "masks" else np.float32), name
        assert shard.meta["channels"][name] == list(spec.tensor(name).channel_names)
    assert shard.array("slot_id").dtype == np.int64
    assert shard.array("step").dtype == np.int64
    assert shard.array("decision_index").shape == (n, 9)
    assert set(shard.meta["missing"]) == {"road_context", "image"}
    assert not shard.has("road_context")
    assert set(shard.meta["excluded"]) == {"eval_only", "privileged_route"}
    assert shard.meta["schema"] == "drivedit.features_v1-shard/1"
    assert shard.meta["split"] == "train"
    assert shard.meta["build_key"] == golden_session.build_key()


def test_channels_equal_the_simulation_truth(golden_shard_dir: Path, golden_truth: dict[str, Any]) -> None:
    """golden.json's decision samples come from make_synthetic's simulation (the command decided at each decision
    callback and where it was applied), not from the reader."""
    shard = _shard(golden_shard_dir)
    truth = golden_truth["decision_steps"]
    assert [d["decision_step"] for d in truth] == shard.array("step").tolist()
    for name in ("state", "action_cmd", "action_label"):
        want = np.array([[math.nan if v is None else v for v in (d[name] or [None] * 3)] for d in truth])
        np.testing.assert_allclose(shard.array(name), want, atol=TOL, equal_nan=True, err_msg=name)
    exec_want = np.full((len(truth), 6, 4), np.nan)
    for t, d in enumerate(truth):
        for k, row in enumerate(d["action_exec"]):
            if row is not None:
                exec_want[t, k] = [math.nan if v is None else v for v in row]
    np.testing.assert_allclose(shard.array("action_exec"), exec_want, atol=TOL, equal_nan=True)
    names = shard.meta["channels"]["masks"]
    masks_want = np.array([[d["masks"][m] for m in names] for d in truth])
    np.testing.assert_array_equal(shard.array("masks"), masks_want)  # nav_valid is 1 on the whole golden route
    slots = [d["obs_slot_id"] if d["obs_valid"] == 1 else -1 for d in truth]
    assert shard.array("slot_id").tolist() == slots
    columns = shard.meta["decision_index_columns"]
    index = shard.array("decision_index")
    want_index = np.array([[math.nan if d[c] is None else d[c] for c in columns] for d in truth], dtype=np.float64)
    np.testing.assert_array_equal(index, want_index)


def test_contract_offsets_against_the_raw_rows(golden_shard_dir: Path, golden_session: Any) -> None:
    """action_cmd_t is ctl.hl_* of the row after C_t, action_label_t ctl.exp_*_now of C_t, state_t the telemetry
    row of C_t (dc-2.0 section 12, the streams JSON timing.sample row offsets)."""
    shard = _shard(golden_shard_dir)
    ctl = golden_session.streams["ctl"]
    tel = golden_session.streams["telemetry"]
    ctl_index, tel_index = ctl.index(), tel.index()
    cmd, label, state = shard.array("action_cmd"), shard.array("action_label"), shard.array("state")
    wm = shard.array("masks")[:, shard.meta["channels"]["masks"].index("wm_valid")]
    differs_from_row_0 = 0
    for t, p in enumerate(shard.array("step").tolist()):
        c0 = ctl_index[p]
        if wm[t] == 1:
            row1 = [ctl.data[k][c0 + 1] for k in ("hl_kappa", "hl_accel", "hl_dir")]
            np.testing.assert_allclose(cmd[t], row1, atol=TOL)
            row0 = [ctl.data[k][c0] for k in ("hl_kappa", "hl_accel", "hl_dir")]
            differs_from_row_0 += any(v is None or abs(a - v) > TOL for a, v in zip(cmd[t], row0, strict=True))
        row = [ctl.data[k][c0] for k in ("exp_kappa_now", "exp_accel_now", "exp_dir_now")]
        if all(v is not None for v in row):
            np.testing.assert_allclose(label[t], row, atol=TOL)
        i = tel_index[p]
        assert abs(state[t, 0] - tel.data["vel_local_z"][i]) < TOL
        assert abs(state[t, 2] - tel.data["acc_local_z"][i]) < TOL
    assert differs_from_row_0 > 0  # offset 1, not 0: C_t's own row still holds the previous command


def test_episodes_segments_and_windows_never_cross(golden_shard_dir: Path) -> None:
    shard = _shard(golden_shard_dir)
    assert [(e.episode_id, e.start, e.stop) for e in shard.episodes] == [(0, 10, 37), (1, 39, 66)]
    assert [(s.episode_id, s.start, s.stop) for s in shard.segments] == [(0, 10, 36), (1, 39, 65)]
    names = shard.meta["channels"]["masks"]
    masks = shard.array("masks")
    episode = np.full(shard.n_steps, -1)
    for e in shard.episodes:
        episode[e.start : e.stop] = e.episode_id
    for length in (1, 5, 26):
        index = WindowIndex([shard], length)
        assert len(index) == sum(max(s.length - length + 1, 0) for s in shard.segments)
        for w in index:
            rows = masks[w.start : w.stop]
            assert (rows[:, names.index("wm_valid")] == 1).all()
            assert (rows[:, names.index("warmup")] == 0).all()
            assert len(set(episode[w.start : w.stop].tolist())) == 1
            assert (shard.array("slot_id")[w.start : w.stop] >= 0).all()
    assert len(WindowIndex([shard], 27)) == 0


def test_nav_invalid_clears_masks_and_splits_segments(
    golden_session: Any, spec: FeatureSpec, tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = build._nav

    def nav_off(*args: Any) -> Any:
        nav, valid, progress = real(*args)
        valid = valid.copy()
        valid[15:18] = 0  # as if the ego were more than 30 m off the route
        return nav, valid, progress

    monkeypatch.setattr(build, "_nav", nav_off)
    shard = _shard(build.write_shard(golden_session, tmp, spec=spec, routes_dir=ROUTES_DIR))
    names = shard.meta["channels"]["masks"]
    masks = shard.array("masks")
    assert (masks[15:18, names.index("policy_valid")] == 0).all()
    assert (masks[15:18, names.index("wm_valid")] == 0).all()
    assert (masks[10:15, names.index("wm_valid")] == 1).all()
    assert [(s.start, s.stop) for s in shard.segments] == [(10, 15), (18, 36), (39, 65)]
    assert shard.meta["counts"]["nav_invalid_in_route_episodes"] == 3


def test_nav_follows_nav_v1(golden_shard_dir: Path, golden_session: Any) -> None:
    nav = build.nav_module()
    shard = _shard(golden_shard_dir)
    route = nav.load_route(str(ROUTES_DIR / "golden_s_curve_v1.json"))
    tel = golden_session.streams["telemetry"]
    index = tel.index()
    heading = golden_session.telemetry_export["heading_rad"]
    for e in shard.episodes:
        steps = shard.array("step")[e.start : e.stop].tolist()
        poses = [(tel.data["pos_x"][index[p]], tel.data["pos_z"][index[p]], float(heading[index[p]])) for p in steps]
        want = [nav.nav_vector(r) for r in nav.nav_sequence(route, poses)]
        np.testing.assert_allclose(shard.array("nav")[e.start : e.stop], want, atol=TOL)
    assert np.isnan(shard.array("nav")[:10]).all()  # warm-up: no episode, no route


def test_future_pose_and_anchors(golden_shard_dir: Path, golden_session: Any) -> None:
    shard = _shard(golden_shard_dir)
    fut = shard.array("future_pose")
    tel = golden_session.streams["telemetry"]
    index = tel.index()
    heading = golden_session.telemetry_export["heading_rad"]
    steps = shard.array("step").tolist()

    def pose(t: int) -> tuple[float, float, float]:
        i = index[steps[t]]
        return tel.data["pos_x"][i], tel.data["pos_z"][i], float(heading[i])

    for t, k in ((10, 1), (12, 5), (20, 16)):
        x0, z0, h0 = pose(t)
        x1, z1, h1 = pose(t + k)
        dx, dz = x1 - x0, z1 - z0
        want = [dx * math.sin(h0) + dz * math.cos(h0), dx * math.cos(h0) - dz * math.sin(h0), h1 - h0]
        np.testing.assert_allclose(fut[t, k - 1], want, atol=TOL)
    # the S-curve: while the ego turns right (yaw_rate > 0) the heading change ahead is right-positive too
    yaw = shard.array("state")[:, 1]
    turning = [t for t in range(10, 30) if yaw[t] > 0.05 and yaw[t + 1] > 0.05]
    assert turning
    assert all(fut[t, 0, 2] > 0 for t in turning)
    # never across an episode end (37) or the teleport between the episodes
    assert np.isnan(fut[36, 0]).all()
    assert np.isfinite(fut[35, 0]).all()
    assert np.isnan(fut[30, 10:]).all()
    assert np.isnan(fut[:10]).all()
    anchors = shard.array("hindsight_anchors")
    assert shard.meta["hindsight_anchor_steps"] == [10, 20, 30]
    np.testing.assert_array_equal(anchors[:, 0:2], fut[:, 9, :2])
    np.testing.assert_array_equal(anchors[:, 2:4], fut[:, 19, :2])
    np.testing.assert_array_equal(anchors[:, 4:6], fut[:, 29, :2])


def test_style_per_episode(golden_shard_dir: Path) -> None:
    shard = _shard(golden_shard_dir)
    style = shard.array("style")
    for e in shard.episodes:
        np.testing.assert_allclose(style[e.start : e.stop], [[1.0, 0.0, 2.75]] * (e.stop - e.start))
    assert np.isnan(style[:10]).all()
    assert np.isnan(style[37:39]).all()


def test_contact_events_from_a_crash(synth: ModuleType, spec: FeatureSpec, tmp: Path) -> None:
    out = synth.generate(str(tmp / "crash"), variant="crash")
    session = build.open_dc2_session(out["session_dir"], vehicle_card=CARD)
    shard = _shard(build.write_shard(session, tmp / "root", spec=spec, routes_dir=ROUTES_DIR))
    collision = next(c for c in session.crash_events if c["kind"] == "collision")
    steps = shard.array("step")
    contacts = shard.array("contact_events")
    hit = np.flatnonzero(contacts[:, 0] == 1).tolist()
    counts = shard.meta["counts"]
    assert counts["ego_collision_rows"] == 1
    assert counts["ego_collision_rows_outside_decision_intervals"] == 0
    (t,) = hit
    assert steps[t] < collision["step"] <= steps[t + 1]  # the decision interval (P_t, P_t+1]
    assert shard.meta["categories"]["contact_class"][int(contacts[t, 1])] == collision["values"]["other"] == "rock"
    assert (contacts[contacts[:, 0] == 0, 1] == -1).all()
    names = shard.meta["channels"]["masks"]
    masks = shard.array("masks")
    # 9.5.1: no policy label from the interval that contains the onset on; crash dynamics stay world-model data
    assert (masks[t:, names.index("policy_valid")] == 0).all()
    assert masks[t, names.index("wm_valid")] == 1
    assert masks[t - 1, names.index("policy_valid")] == 1


def test_the_10hz_override(synth: ModuleType, spec: FeatureSpec, tmp: Path, golden_shard_dir: Path) -> None:
    out = synth.generate(str(tmp / "s"), kind="golden_10hz")
    session = build.open_dc2_session(out["session_dir"], vehicle_card=CARD)
    shard = _shard(build.write_shard(session, tmp / "root", spec=spec, routes_dir=ROUTES_DIR))
    n = shard.n_steps
    assert shard.decision_hz == 10
    assert shard.array("action_exec").shape == (n, 3, 4)
    assert shard.array("future_pose").shape == (n, 60, 3)
    assert shard.meta["hindsight_anchor_steps"] == [20, 40, 60]
    with pytest.raises(ValueError, match="decision rates"):
        WindowIndex([shard, _shard(golden_shard_dir)], 4)


def test_deterministic_bytes(golden_session: Any, spec: FeatureSpec, tmp: Path, golden_shard_dir: Path) -> None:
    again = build.write_shard(golden_session, tmp, spec=spec, routes_dir=ROUTES_DIR)
    files = sorted(p.name for p in golden_shard_dir.iterdir())
    assert files == sorted(p.name for p in again.iterdir())
    for name in files:
        assert (golden_shard_dir / name).read_bytes() == (again / name).read_bytes(), name


def test_overwrite_and_output_refusals(golden_session: Any, spec: FeatureSpec, tmp: Path) -> None:
    first = build.write_shard(golden_session, tmp, spec=spec, routes_dir=ROUTES_DIR)
    with pytest.raises(build.BuildError, match="exists"):
        build.write_shard(golden_session, tmp, spec=spec, routes_dir=ROUTES_DIR)
    assert build.write_shard(golden_session, tmp, spec=spec, routes_dir=ROUTES_DIR, overwrite=True) == first
    assert sorted(p.name for p in first.parent.iterdir()) == ["features_v1"]  # no temporary left behind
    with pytest.raises(build.BuildError, match="inside"):
        build.write_shard(golden_session, GOLDEN_DIR / "out", spec=spec, routes_dir=ROUTES_DIR)
    with pytest.raises(build.BuildError, match="C: drive"):
        build.write_shard(golden_session, "/mnt/c/dd11", spec=spec, routes_dir=ROUTES_DIR)


def test_image_derivatives(spec: FeatureSpec, tmp: Path) -> None:
    session_dir = tmp / "captures" / "arland" / GOLDEN_DIR.name
    shutil.copytree(GOLDEN_DIR, session_dir)
    deriv = session_dir / "_harvest" / "deriv"
    deriv.mkdir(parents=True)
    for slot in range(68):
        (deriv / f"f_{slot:08d}.jpg").write_bytes(b"")
    session = build.open_dc2_session(session_dir, vehicle_card=CARD)
    with pytest.raises(build.BuildError, match="inside"):
        build.write_shard(session, tmp / "captures" / "datasets", spec=spec, routes_dir=ROUTES_DIR)
    shard = _shard(build.write_shard(session, tmp / "datasets", spec=spec, routes_dir=ROUTES_DIR))
    assert "image" not in shard.meta["missing"]
    assert shard.meta["image"]["pattern"] == "f_{slot:08d}.jpg"
    slot = int(shard.array("slot_id")[20])
    assert shard.image_path(slot) == deriv.resolve() / f"f_{slot:08d}.jpg"
    (deriv / f"f_{slot:08d}.jpg").unlink()
    with pytest.raises(build.BuildError, match="derivative"):
        build.write_shard(session, tmp / "datasets2", spec=spec, routes_dir=ROUTES_DIR)


def test_route_provenance(golden_session: Any, spec: FeatureSpec, tmp: Path) -> None:
    obj = json.loads((ROUTES_DIR / "golden_s_curve_v1.json").read_text("utf-8"))
    obj["polyline"] = [[x + 1.0, y, z] for x, y, z in obj["polyline"]]
    routes = tmp / "routes"
    routes.mkdir()
    (routes / "golden_s_curve_v1.json").write_text(json.dumps(obj), "utf-8")
    with pytest.raises(build.BuildError, match="route sha256"):
        build.write_shard(golden_session, tmp / "a", spec=spec, routes_dir=routes)
    with pytest.raises(build.BuildError, match="no route file"):
        build.write_shard(golden_session, tmp / "b", spec=spec, routes_dir=tmp)


def test_logged_route_hash_forms(tmp: Path) -> None:
    """The 8191-character prefix form of route_sha256 (the pre-cc31f07 Substring defect) identifies the same route
    only when the route is longer than that."""
    nav = build.nav_module()
    long_route = build.default_routes_dir() / "arland_t0_loop.json"
    route = nav.load_route(str(long_route))
    text = nav.canonical_json(json.loads(long_route.read_text("utf-8"))).encode("ascii")
    assert len(text) > 8191
    assert build._logged_route_hash(route.sha256, route, long_route) == "exact"
    assert build._logged_route_hash(hashlib.sha256(text[:8191]).hexdigest(), route, long_route) == "prefix_8191"
    assert build._logged_route_hash(hashlib.sha256(text[:8190]).hexdigest(), route, long_route) is None
    short = ROUTES_DIR / "golden_s_curve_v1.json"
    short_route = nav.load_route(str(short))
    short_text = nav.canonical_json(json.loads(short.read_text("utf-8"))).encode("ascii")
    assert len(short_text) <= 8191
    assert build._logged_route_hash(hashlib.sha256(short_text[:100]).hexdigest(), short_route, short) is None


def test_a_session_without_a_vehicle_card_is_refused(spec: FeatureSpec, tmp: Path) -> None:
    session = build.open_dc2_session(GOLDEN_DIR)  # no tools/ident/cards/1.8.0.13/S105_synthetic.json
    with pytest.raises(build.BuildError, match="vehicle card"):
        build.write_shard(session, tmp, spec=spec, routes_dir=ROUTES_DIR)


def test_eval_only_needs_its_flag(golden_session: Any, spec: FeatureSpec, tmp: Path) -> None:
    shard = _shard(
        build.write_shard(
            golden_session, tmp, spec=spec, routes_dir=ROUTES_DIR, include_eval_only=True, include_privileged=True
        )
    )
    assert shard.meta["channels"]["eval_only"] == list(build.EVAL_ONLY_BUILT)
    eval_only = shard.array("eval_only")
    assert eval_only.shape == (68, len(build.EVAL_ONLY_BUILT))
    assert "privileged_route" in shard.meta["missing"]
    assert shard.meta["excluded"] == {}
    tel = golden_session.streams["telemetry"]
    i = tel.index()[int(shard.array("step")[20])]
    assert abs(eval_only[20, 0] - tel.data["pos_x"][i]) < 1e-2
    speed = math.hypot(float(eval_only[20, 4]), float(eval_only[20, 6]))
    assert abs(speed - float(shard.array("state")[20, 0])) < 1e-3  # world velocity = the rotated local one


def test_cli(tmp: Path) -> None:
    code = build.main(
        [
            "--session",
            str(GOLDEN_DIR),
            "--out",
            str(tmp),
            "--routes-dir",
            str(ROUTES_DIR),
            "--vehicle-card",
            str(CARD),
        ]
    )
    assert code == 0
    assert Shard(tmp / GOLDEN_DIR.name / "features_v1").n_steps == 68
