"""drivedit.data.build on the golden synthetic dc-2.0 session: shapes and dtypes per features_v1, the contract
offsets (against the raw ctl/telemetry rows and against golden.json, which make_synthetic.py computed from its own
simulation), masks and segments, future_pose and the anchors, contact_events, the image source, refusals, the 10 Hz
override and determinism."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from drivedit.data import build
from drivedit.data.shard import Shard, WindowIndex
from drivedit.data.spec import FeatureSpec, session_split

from .conftest import CARD, GOLDEN_DIR, REPO, ROUTES_DIR

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
    assert shard.meta["route_split"] == "train"
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
    datasets = tmp / "datasets"
    shard = _shard(
        build.write_shard(golden_session, datasets, spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(datasets,))
    )
    names = shard.meta["channels"]["masks"]
    masks = shard.array("masks")
    assert (masks[15:18, names.index("policy_valid")] == 0).all()
    assert (masks[15:18, names.index("wm_valid")] == 0).all()
    assert (masks[10:15, names.index("wm_valid")] == 1).all()
    assert [(s.start, s.stop) for s in shard.segments] == [(10, 15), (18, 36), (39, 65)]
    assert shard.meta["counts"]["nav_invalid_in_route_episodes"] == 3


def _per_row_nav(session: Any, route: Any, episode_id: int) -> dict[int, dict[str, Any]]:
    """The contract's nav view, written out independently of the builder: nav_v1 over every telemetry row of the
    episode in row order (dc-2.0 section 9.8, streams JSON derived.nav; validator C17 walks the rows the same way)."""
    nav = build.nav_module()
    tel = session.streams["telemetry"]
    heading = session.telemetry_export["heading_rad"]
    rows = [i for i, e in enumerate(tel.data["episode_id"]) if e == episode_id]
    poses = [(tel.data["pos_x"][i], tel.data["pos_z"][i], float(heading[i])) for i in rows]
    return {tel.data["physics_step"][i]: r for i, r in zip(rows, nav.nav_sequence(route, poses), strict=True)}


def test_nav_is_nav_v1_over_every_telemetry_row(golden_shard_dir: Path, golden_session: Any) -> None:
    nav = build.nav_module()
    shard = _shard(golden_shard_dir)
    route = nav.load_route(str(ROUTES_DIR / "golden_s_curve_v1.json"))
    for e in shard.episodes:
        per_row = _per_row_nav(golden_session, route, e.episode_id)
        steps = shard.array("step")[e.start : e.stop].tolist()
        want = [nav.nav_vector(per_row[p]) for p in steps]
        np.testing.assert_allclose(shard.array("nav")[e.start : e.stop], want, atol=TOL)
    assert np.isnan(shard.array("nav")[:10]).all()  # warm-up: no episode, no route


def _straight_route() -> Any:
    """A 100 m route due north (+z) from the origin: progress is the ego's z."""
    obj = {
        "schema": "drivedit.route/1",
        "route_id": "straight_north_v1",
        "map_id": "arland",
        "split": "train",
        "nav_version": "nav_v1",
        "lane_side": "right",
        "closed": False,
        "polyline": [[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]],
        "maneuvers": [],
        "stops": [{"s_m": 100.0, "kind": "route_end"}],
    }
    return build.nav_module().route_from_dict(obj)


def test_progress_carries_through_rows_between_decisions() -> None:
    """The counterexample of review round 1: decisions at steps 1 and 3, and a row between them that advances to
    11 m before the ego is back at 10 m. Per telemetry row (the contract) progress is 11 m at the second decision
    and p20 lies 21 m ahead; evaluating the decision poses only would give 10 m and 20 m."""
    nav = build.nav_module()
    route = _straight_route()
    rows = [(1, (0.0, 0.0, 0.0)), (2, (0.0, 11.0, 0.0)), (3, (0.0, 10.0, 0.0))]
    at = build.episode_nav(route, rows)
    assert at[3]["progress_m"] == pytest.approx(11.0)
    assert at[3]["p20_x"] == pytest.approx(21.0)
    subsampled = nav.nav_sequence(route, [rows[0][1], rows[2][1]])
    assert subsampled[1]["progress_m"] == pytest.approx(10.0)
    assert subsampled[1]["p20_x"] == pytest.approx(20.0)


def _rewrite_telemetry_row(session_dir: Path, physics_step: int, values: dict[str, float]) -> None:
    """Change fields of one committed telemetry row in place and keep its chunk's commit line (n_bytes) true."""
    sdir = session_dir / "telemetry"
    commit = (sdir / "commit.csv").read_text("ascii").split("\n")
    header = commit[0].split(",")
    for n, line in enumerate(commit[1:-1], start=1):
        c = dict(zip(header, line.split(","), strict=True))
        if int(c["first_key"]) <= physics_step <= int(c["last_key"]):
            path = sdir / f"chunk_{int(c['chunk_id']):05d}.csv"
            lines = path.read_text("ascii").split("\n")
            cols = lines[0].split(",")
            for i, row in enumerate(lines[1:-1], start=1):
                fields = row.split(",")
                if int(fields[0]) == physics_step:
                    for name, v in values.items():
                        fields[cols.index(name)] = f"{v:.4f}"
                    lines[i] = ",".join(fields)
            data = "\n".join(lines).encode("ascii")
            path.write_bytes(data)
            c["n_bytes"] = str(len(data))
            commit[n] = ",".join(c[h] for h in header)
            (sdir / "commit.csv").write_text("\n".join(commit), "ascii")
            return
    raise AssertionError(f"no telemetry chunk holds physics step {physics_step}")


def test_nav_uses_the_rows_between_decisions(golden_session: Any, spec: FeatureSpec, tmp: Path) -> None:
    """End to end: a telemetry row between two decision callbacks moved 3 m ahead along the route moves the next
    decision's progress and route points; the shard follows the per-row view, not the decision poses alone."""
    nav = build.nav_module()
    route = nav.load_route(str(ROUTES_DIR / "golden_s_curve_v1.json"))
    steps = [int(d["decision_step"]) for d in golden_session.decisions]
    t = 21
    tel = golden_session.streams["telemetry"]
    between = [p for p in tel.data["physics_step"] if steps[t - 1] < p < steps[t]]
    assert between
    i = tel.index()[steps[t]]
    h = float(golden_session.telemetry_export["heading_rad"][i])
    x, z = tel.data["pos_x"][i] + 3.0 * math.sin(h), tel.data["pos_z"][i] + 3.0 * math.cos(h)
    session_dir = tmp / "s" / GOLDEN_DIR.name
    shutil.copytree(GOLDEN_DIR, session_dir)
    _rewrite_telemetry_row(session_dir, between[-1], {"pos_x": x, "pos_z": z})
    moved = build.open_dc2_session(session_dir, vehicle_card=CARD)
    datasets = tmp / "datasets"
    shard = _shard(
        build.write_shard(
            moved, datasets, spec=spec, routes_dir=ROUTES_DIR, include_eval_only=True, dataset_roots=(datasets,)
        )
    )
    per_row = _per_row_nav(moved, route, 0)
    np.testing.assert_allclose(shard.array("nav")[t], nav.nav_vector(per_row[steps[t]]), atol=TOL)
    progress = shard.array("eval_only")[:, list(build.EVAL_ONLY_BUILT).index("route_progress_m")]
    m_tel = moved.streams["telemetry"]
    m_heading = moved.telemetry_export["heading_rad"]
    decision_only = nav.nav_sequence(
        route,
        [
            (
                m_tel.data["pos_x"][m_tel.index()[p]],
                m_tel.data["pos_z"][m_tel.index()[p]],
                float(m_heading[m_tel.index()[p]]),
            )
            for p in steps[10 : t + 1]
        ],
    )
    assert progress[t] > decision_only[-1]["progress_m"] + 2.0
    assert abs(float(shard.array("nav")[t, 0]) - decision_only[-1]["p20_x"]) > 1.0


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


def _timeline(episode: list[int], grid: list[int], tele_steps: tuple[int, ...] = ()) -> Any:
    """A synthetic decision timeline: one decision every 6 physics steps, the ego 2 m further north at each."""
    n = len(episode)
    return build._Timeline(
        steps=np.arange(n, dtype=np.int64) * 6,
        grid=np.array(grid, dtype=np.int64),
        episode=np.array(episode, dtype=np.int64),
        poses=np.array([[0.0, 2.0 * t, 0.0] for t in range(n)]),
        tele_steps=tele_steps,
        decision_hz=5.0,
        n_exec=6,
        horizon=3,
    )


def test_pose_deltas_stop_at_every_boundary() -> None:
    """future_pose never spans two episodes (even back to back, with no teleport between them), a teleport or
    respawn inside an episode, or a skipped grid point. On the golden session the teleport between the episodes
    would hide a missing episode check, so the checks are tested one at a time here."""
    k = [1, 2, 3]
    adjacent = build._ego_deltas(_timeline([0, 0, 0, 1, 1, 1], list(range(6))), k)
    np.testing.assert_allclose(adjacent[0, :2], [[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    assert np.isnan(adjacent[0, 2]).all()  # t=0 + 3 is in episode 1
    assert np.isnan(adjacent[2]).all()  # the last step of episode 0
    teleported = build._ego_deltas(_timeline([0] * 6, list(range(6)), tele_steps=(13,)), k)
    assert np.isfinite(teleported[0, 1]).all()  # (0, 12] holds no teleport
    assert np.isnan(teleported[0, 2]).all()  # (0, 18] holds step 13
    assert np.isnan(teleported[2, 0]).all()  # (12, 18]
    assert np.isfinite(teleported[3, :2]).all()  # after the teleport
    skipped = build._ego_deltas(_timeline([0] * 6, [0, 1, 2, 4, 5, 6]), k)
    assert np.isfinite(skipped[0, :2]).all()
    assert np.isnan(skipped[0, 2]).all()  # three decisions later is four grid points later
    assert np.isfinite(skipped[3, :2]).all()


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
    datasets = tmp / "datasets"
    shard = _shard(build.write_shard(session, datasets, spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(datasets,)))
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
    datasets = tmp / "datasets"
    shard = _shard(build.write_shard(session, datasets, spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(datasets,)))
    n = shard.n_steps
    assert shard.decision_hz == 10
    assert shard.array("action_exec").shape == (n, 3, 4)
    assert shard.array("future_pose").shape == (n, 60, 3)
    assert shard.meta["hindsight_anchor_steps"] == [20, 40, 60]
    with pytest.raises(ValueError, match="decision rates"):
        WindowIndex([shard, _shard(golden_shard_dir)], 4)


def test_deterministic_bytes(golden_session: Any, spec: FeatureSpec, tmp: Path, golden_shard_dir: Path) -> None:
    again = build.write_shard(golden_session, tmp, spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(tmp,))
    files = sorted(p.name for p in golden_shard_dir.iterdir())
    assert files == sorted(p.name for p in again.iterdir())
    for name in files:
        assert (golden_shard_dir / name).read_bytes() == (again / name).read_bytes(), name


def test_overwrite_and_output_refusals(golden_session: Any, spec: FeatureSpec, tmp: Path) -> None:
    datasets = tmp / "datasets"
    kw: dict[str, Any] = {"spec": spec, "routes_dir": ROUTES_DIR, "dataset_roots": (datasets,)}
    first = build.write_shard(golden_session, datasets, **kw)
    with pytest.raises(build.BuildError, match="exists"):
        build.write_shard(golden_session, datasets, **kw)
    assert build.write_shard(golden_session, datasets, overwrite=True, **kw) == first
    assert sorted(p.name for p in first.parent.iterdir()) == ["features_v1"]  # no temporary left behind
    with pytest.raises(build.BuildError, match="not under a dataset root"):
        build.write_shard(golden_session, tmp / "elsewhere", **kw)
    with pytest.raises(build.BuildError, match="overlaps"):  # a dataset root that holds the session
        build.write_shard(
            golden_session, GOLDEN_DIR / "out", spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(GOLDEN_DIR,)
        )
    with pytest.raises(build.BuildError, match="C: drive"):
        build.write_shard(golden_session, "/mnt/c/dd11", spec=spec, routes_dir=ROUTES_DIR, dataset_roots=("/mnt/c",))
    with pytest.raises(build.BuildError, match="no dataset root"):
        build.write_shard(golden_session, datasets, spec=spec, routes_dir=ROUTES_DIR, dataset_roots=())
    assert not (tmp / "elsewhere").exists()


def _tree(path: Path) -> list[Path]:
    return sorted(p.relative_to(path) for p in path.rglob("*"))


def test_a_normal_build_goes_under_an_allowed_root(
    spec: FeatureSpec, tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    datasets = tmp / "datasets"
    shard = build.build_session(
        GOLDEN_DIR, datasets / "a", spec=spec, routes_dir=ROUTES_DIR, vehicle_card=CARD, dataset_roots=(datasets,)
    )
    assert shard == (datasets / "a" / GOLDEN_DIR.name / "features_v1").resolve()
    assert Shard(shard).n_steps == 68
    monkeypatch.setenv(build.DATASET_ROOTS_ENV, str(datasets))  # the variable replaces the default root
    shard = build.build_session(GOLDEN_DIR, datasets / "b", spec=spec, routes_dir=ROUTES_DIR, vehicle_card=CARD)
    assert Shard(shard).n_steps == 68


def test_probe_dotdot_through_a_symlink(spec: FeatureSpec, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 3, probe 1, unmocked: portal -> nested/deeper and nested/captures -> storage, so
    portal/../captures/<id> opens storage/<id> while abspath() would name <tmp>/captures. An out root under
    nested/captures/ resolves into storage and lies under no dataset root: refused, nothing written."""
    storage = tmp / "storage"
    shutil.copytree(GOLDEN_DIR, storage / GOLDEN_DIR.name)
    (tmp / "nested" / "deeper").mkdir(parents=True)
    (tmp / "portal").symlink_to(tmp / "nested" / "deeper")
    (tmp / "nested" / "captures").symlink_to(storage)
    monkeypatch.chdir(tmp)
    named = Path("portal") / ".." / "captures" / GOLDEN_DIR.name
    assert named.resolve() == (storage / GOLDEN_DIR.name).resolve()
    before = _tree(storage)
    kw: dict[str, Any] = {"spec": spec, "routes_dir": ROUTES_DIR, "vehicle_card": CARD}
    for allowed in ((tmp / "datasets",), (tmp / "nested",), (tmp,)):  # (tmp,) holds the session itself
        with pytest.raises(build.BuildError, match="refusing"):
            build.build_session(named, Path("nested") / "captures" / "new-output", dataset_roots=allowed, **kw)
    assert _tree(storage) == before
    assert _tree(tmp / "nested") == [Path("captures"), Path("deeper")]


def test_probe_write_shard_after_opening_through_a_symlinked_root(spec: FeatureSpec, tmp: Path) -> None:
    """Review round 3, probe 2, unmocked: with captures -> storage, write_shard(open_dc2_session(captures/<id>),
    captures/new-output) needs no session path as named: the out root is under no dataset root."""
    storage = tmp / "storage"
    shutil.copytree(GOLDEN_DIR, storage / GOLDEN_DIR.name)
    (tmp / "captures").symlink_to(storage)
    session = build.open_dc2_session(tmp / "captures" / GOLDEN_DIR.name, vehicle_card=CARD)
    before = _tree(storage)
    for allowed in ((tmp / "datasets",), (tmp,)):
        with pytest.raises(build.BuildError, match="refusing"):
            build.write_shard(
                session, tmp / "captures" / "new-output", spec=spec, routes_dir=ROUTES_DIR, dataset_roots=allowed
            )
    assert _tree(storage) == before
    shard = build.write_shard(
        session, tmp / "datasets", spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(tmp / "datasets",)
    )
    assert Shard(shard).n_steps == 68
    assert _tree(storage) == before


def test_probe_empty_or_partial_configuration_keeps_the_defaults(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 3, probe 3: an empty or blank variable counts as unset, and capture roots only add, so neither
    can drop /mnt/e/DriveDiT/captures; an out root there lies under no default dataset root."""
    default_captures = [Path(p) for p in build.CAPTURE_ROOTS]
    default_datasets = [Path(p) for p in build.DATASET_ROOTS]
    assert default_captures == [Path("/mnt/e/DriveDiT/captures")]
    assert default_datasets == [Path("/mnt/e/DriveDiT/datasets")]
    session = SimpleNamespace(path=str(tmp / "s" / GOLDEN_DIR.name), session_id=GOLDEN_DIR.name)
    for value in ("", " ", os.pathsep, f" {os.pathsep} "):
        monkeypatch.setenv(build.CAPTURE_ROOTS_ENV, value)
        monkeypatch.setenv(build.DATASET_ROOTS_ENV, value)
        assert build.configured_capture_roots() == default_captures
        assert build.allowed_dataset_roots() == default_datasets
        with pytest.raises(build.BuildError, match="not under a dataset root"):
            build.write_shard(session, "/mnt/e/DriveDiT/captures/new-output", spec=None)
        # a dataset root that holds the capture store is refused, whatever else is configured
        for extra in ((), (tmp / "other_store",)):
            with pytest.raises(build.BuildError, match="overlaps /mnt/e/DriveDiT/captures"):
                build.write_shard(
                    session,
                    "/mnt/e/DriveDiT/captures/new-output",
                    spec=None,
                    dataset_roots=("/mnt/e/DriveDiT",),
                    capture_roots=extra,
                )
    monkeypatch.setenv(build.CAPTURE_ROOTS_ENV, str(tmp / "env_store"))
    assert build.configured_capture_roots([tmp / "kw_store"]) == [
        *default_captures,
        tmp / "env_store",
        tmp / "kw_store",
    ]
    store = tmp / "kw_store"
    store.mkdir()
    (tmp / "store_link").symlink_to(store)
    for out_root, allowed in ((store / "x", (store,)), (tmp / "store_link" / "x", (tmp / "store_link",))):
        with pytest.raises(build.BuildError, match="overlaps"):
            build.write_shard(session, out_root, spec=None, dataset_roots=allowed, capture_roots=[tmp / "store_link"])
    assert list(store.iterdir()) == []


def test_containment_is_decided_by_filesystem_identity(tmp: Path) -> None:
    """Review round 4: on drvfs /mnt/e/drivedit and /mnt/e/DriveDiT are one directory while resolve() keeps each
    spelling. tmpfs is case-sensitive, so an unresolved symlink alias stands in for the second spelling here: only
    (st_dev, st_ino) can tell that alias/x lies in store, also for an output that does not exist yet."""
    store = tmp / "store"
    (store / "sub").mkdir(parents=True)
    alias = tmp / "alias"
    alias.symlink_to(store)
    assert not (alias / "new" / "x").is_relative_to(store)  # the strings disagree
    assert build._inside(alias / "new" / "x", store)  # a path that does not exist yet
    assert build._inside(alias / "sub", store)
    assert build._inside(store / "sub", alias)
    assert build._inside(alias, store)
    assert not build._inside(tmp / "elsewhere" / "x", store)
    assert not build._inside(tmp, store)
    assert build._inside(tmp / "missing_root" / "x", tmp / "missing_root")  # no root yet: the string decides


DRVFS_CAPTURES = Path("/mnt/e/DriveDiT/captures")


@pytest.mark.skipif(
    not (DRVFS_CAPTURES.is_dir() and Path("/mnt/e/drivedit/captures").is_dir()),
    reason="needs the case-insensitive /mnt/e (drvfs) capture store",
)
def test_drvfs_case_alias_of_the_capture_store_is_refused(tmp: Path) -> None:
    """The real case alias, read only: the guard refuses before anything is computed (a stand-in session, so even a
    wrong decision could write nothing)."""
    assert os.path.samefile(DRVFS_CAPTURES, "/mnt/e/drivedit/captures")
    assert Path("/mnt/e/drivedit/captures").resolve() != DRVFS_CAPTURES  # resolve() keeps the spelling
    session = SimpleNamespace(path=str(DRVFS_CAPTURES / "arland" / "s" / GOLDEN_DIR.name), session_id=GOLDEN_DIR.name)
    for alias in ("/mnt/e/drivedit/captures", "/mnt/e/DRIVEDIT/CAPTURES", "/mnt/e/drivedit"):
        with pytest.raises(build.BuildError, match="overlaps"):
            build.write_shard(session, Path(alias) / "new-output", spec=None, dataset_roots=(alias,))
    assert not (DRVFS_CAPTURES / "new-output").exists()


@pytest.mark.parametrize(
    "value",
    ["E:/DriveDiT/datasets", "E:\\DriveDiT\\datasets", "e:/x", "E:", "E:relative", "/mnt/e/x\\y"],
)
def test_windows_spellings_are_refused(value: str, tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Review round 4: E:/x splits on ':' into E and /x, and backslashes name other paths, so a root given in Windows
    spelling would protect or allow nothing it names. Refused in the variables (also inside a list), the keywords
    and the CLI, with the WSL spelling in the message."""
    session = SimpleNamespace(path=str(tmp / "s" / GOLDEN_DIR.name), session_id=GOLDEN_DIR.name)
    datasets = tmp / "datasets"
    for name in (build.CAPTURE_ROOTS_ENV, build.DATASET_ROOTS_ENV):
        for raw in (value, f"{tmp / 'a'}{os.pathsep}{value}"):
            monkeypatch.setenv(name, raw)
            with pytest.raises(build.BuildError, match="/mnt/<drive>"):
                build.write_shard(session, datasets, spec=None)
        monkeypatch.delenv(name)
    with pytest.raises(build.BuildError, match="/mnt/<drive>"):
        build.write_shard(session, datasets, spec=None, dataset_roots=(datasets, value))
    with pytest.raises(build.BuildError, match="/mnt/<drive>"):
        build.write_shard(session, datasets, spec=None, dataset_roots=(datasets,), capture_roots=(value,))
    for flag in ("--dataset-root", "--capture-root"):
        with pytest.raises(build.BuildError, match="/mnt/<drive>"):  # before the session is read
            build.main(
                [
                    "--session",
                    str(tmp / "no_session"),
                    "--out",
                    str(datasets),
                    "--dataset-root",
                    str(datasets),
                    flag,
                    value,
                ]
            )
    assert not datasets.exists()


def _c02_split(session_dir: Path) -> Any:
    """The split validator C02 reports for a session (tools/validator/dc_validate.py, run in process)."""
    module_spec = importlib.util.spec_from_file_location("dc_validate_dd11", REPO / "tools/validator/dc_validate.py")
    assert module_spec is not None
    assert module_spec.loader is not None
    dc_validate = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(dc_validate)
    out = io.StringIO()
    argv = ["--session", str(session_dir), "--routes-dir", str(ROUTES_DIR), "--vehicle-card", str(CARD)]
    with contextlib.redirect_stdout(out):
        dc_validate.main([*argv, "--checks", "C02", "--json", "--no-report"])
    (c02,) = [c for c in json.loads(out.getvalue())["checks"] if c["id"] == "C02"]
    return c02["value"]["split"]


def _renamed(session_dir: Path, session_id: str, parent: Path) -> Path:
    """A copy of a session under another session id (session.json, session_end.json and harvest.json)."""
    dst = parent / session_id
    shutil.copytree(session_dir, dst)
    for name in ("session.json", "session_end.json", "harvest.json"):
        obj = json.loads((dst / name).read_text("utf-8"))
        obj["session_id"] = session_id
        (dst / name).write_text(json.dumps(obj, indent=2) + "\n", "utf-8")
    return dst


def test_effective_split_agrees_with_validator_c02(synth: ModuleType, spec: FeatureSpec, tmp: Path) -> None:
    """Review (DD-20 round 2): meta split is the effective split of dc-2.0 3.2 (val_c is the tenth of the train
    sessions whose sha256(session_id)[:8] is 0 mod 10), route_split the raw tag; session_split and validator C02
    give the same answer on the golden and crash fixtures and on a session id in the val_c bucket."""
    assert session_split("arland_T0b_20261025T091512Z_wb_i00_0004", "train") == "val_c"
    assert session_split("arland_T0b_20261025T091512Z_wb_i00_0004", "val_l") == "val_l"
    assert session_split("arland_T0b_20261025T091512Z_wb_i00_0004", None) is None
    assert session_split(GOLDEN_DIR.name, "train") == "train"
    crash = Path(synth.generate(str(tmp / "crash"), variant="crash")["session_dir"])
    val_c = _renamed(GOLDEN_DIR, "arland_T0b_20261025T091512Z_wb_i00_0004", tmp / "renamed")
    for session_dir, want in ((GOLDEN_DIR, "train"), (crash, "train"), (val_c, "val_c")):
        session = build.open_dc2_session(session_dir, vehicle_card=CARD)
        out = tmp / "datasets" / want / session_dir.parent.name
        shard = Shard(
            build.write_shard(session, out, spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(tmp / "datasets",))
        )
        assert shard.meta["split"] == _c02_split(session_dir) == want, session_dir
        assert shard.meta["route_split"] == "train"
        assert {e["route_split"] for e in shard.meta["episodes"]} == {"train"}


def test_image_derivatives(spec: FeatureSpec, tmp: Path) -> None:
    session_dir = tmp / "captures" / "arland" / GOLDEN_DIR.name
    shutil.copytree(GOLDEN_DIR, session_dir)
    deriv = session_dir / "_harvest" / "deriv"
    deriv.mkdir(parents=True)
    for slot in range(68):
        (deriv / f"f_{slot:08d}.jpg").write_bytes(b"")
    session = build.open_dc2_session(session_dir, vehicle_card=CARD)
    with pytest.raises(build.BuildError, match="not under a dataset root"):
        build.write_shard(
            session, tmp / "captures" / "datasets", spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(tmp / "datasets",)
        )
    shard = _shard(
        build.write_shard(
            session, tmp / "datasets", spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(tmp / "datasets",)
        )
    )
    assert "image" not in shard.meta["missing"]
    assert shard.meta["image"]["pattern"] == "f_{slot:08d}.jpg"
    slot = int(shard.array("slot_id")[20])
    assert shard.image_path(slot) == deriv.resolve() / f"f_{slot:08d}.jpg"
    (deriv / f"f_{slot:08d}.jpg").unlink()
    with pytest.raises(build.BuildError, match="derivative"):
        build.write_shard(
            session, tmp / "datasets" / "2", spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(tmp / "datasets",)
        )


def test_route_provenance(golden_session: Any, spec: FeatureSpec, tmp: Path) -> None:
    obj = json.loads((ROUTES_DIR / "golden_s_curve_v1.json").read_text("utf-8"))
    obj["polyline"] = [[x + 1.0, y, z] for x, y, z in obj["polyline"]]
    routes = tmp / "routes"
    routes.mkdir()
    (routes / "golden_s_curve_v1.json").write_text(json.dumps(obj), "utf-8")
    with pytest.raises(build.BuildError, match="route sha256"):
        build.write_shard(golden_session, tmp / "a", spec=spec, routes_dir=routes, dataset_roots=(tmp,))
    with pytest.raises(build.BuildError, match="no route file"):
        build.write_shard(golden_session, tmp / "b", spec=spec, routes_dir=tmp, dataset_roots=(tmp,))


def test_logged_route_hash_forms(tmp: Path) -> None:
    """The 8191-byte prefix form of route_sha256 (the DD-7 Substring defect) is accepted only for the allow-listed,
    unedited deployed route version (PREFIX_HASH_ROUTES)."""
    nav = build.nav_module()
    deployed = build.default_routes_dir() / "arland_t0_loop.json"
    raw = deployed.read_bytes()
    route = nav.load_route(str(deployed))
    assert route.sha256 == build.PREFIX_HASH_ROUTES["arland_t0_loop"]
    assert len(raw) > 8191
    prefix = hashlib.sha256(raw[:8191]).hexdigest()
    assert prefix.startswith("6370fc78f426")  # the value the first T0L session logged
    assert build._logged_route_hash(route.sha256, route, deployed) == {
        "logged": route.sha256,
        "form": "exact",
        "verified_against": route.sha256,
    }
    assert build._logged_route_hash(prefix, route, deployed) == {
        "logged": prefix,
        "form": "prefix_8191",
        "verified_against": build.PREFIX_HASH_ROUTES["arland_t0_loop"],
    }
    assert build._logged_route_hash(hashlib.sha256(raw[:8190]).hexdigest(), route, deployed) is None

    def edited(obj: dict[str, Any], name: str, *, pretty: bool = False) -> tuple[Any, Path]:
        path = tmp / name / "arland_t0_loop.json"
        path.parent.mkdir()
        text = json.dumps(obj, indent=1) if pretty else nav.canonical_json(obj)
        path.write_text(text + "\n", "ascii")
        return nav.load_route(str(path)), path

    # review round 1: an edit after the first 8191 bytes (the split) keeps the prefix hash but is another route
    obj = json.loads(raw)
    obj["split"] = "val_l"
    split_route, split_path = edited(obj, "split")
    assert hashlib.sha256(split_path.read_bytes()[:8191]).hexdigest() == prefix  # the prefix alone cannot tell
    assert split_route.sha256 != route.sha256
    assert build._logged_route_hash(prefix, split_route, split_path) is None
    # the same content in another layout: the bytes the engine hashed differ
    same_route, same_path = edited(json.loads(raw), "pretty", pretty=True)
    assert same_route.sha256 == route.sha256
    assert build._logged_route_hash(prefix, same_route, same_path) is None
    # a long route that is not allow-listed never matches by prefix
    obj = json.loads(raw)
    obj["route_id"] = "arland_t0_loop_copy"
    other, other_path = edited(obj, "other")
    other_prefix = hashlib.sha256(other_path.read_bytes()[:8191]).hexdigest()
    assert build._logged_route_hash(other_prefix, other, other_path) is None


def test_output_paths_are_checked_after_resolving_symlinks(tmp: Path) -> None:
    """Review round 1: <out_root>/<session_id> (or its features_v1) linked into a session or the capture tree must be
    refused before anything is written, also without overwrite, though the out root lies under a dataset root."""
    session_dir = tmp / "captures" / "arland" / GOLDEN_DIR.name
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text("{}", "ascii")
    session = SimpleNamespace(path=str(session_dir), session_id=GOLDEN_DIR.name)
    datasets = tmp / "datasets"
    linked = datasets / "out_a"
    linked.mkdir(parents=True)
    (linked / GOLDEN_DIR.name).symlink_to(session_dir)
    shard_link = datasets / "out_b" / GOLDEN_DIR.name
    shard_link.mkdir(parents=True)
    (shard_link / "features_v1").symlink_to(session_dir)
    elsewhere = datasets / "out_c"
    elsewhere.mkdir()
    (elsewhere / GOLDEN_DIR.name).symlink_to(tmp / "captures")
    for out_root in (linked, datasets / "out_b", elsewhere, session_dir, tmp / "captures"):
        for overwrite in (False, True):
            with pytest.raises(build.BuildError, match="refusing"):
                build.write_shard(session, out_root, spec=None, overwrite=overwrite, dataset_roots=(datasets,))
    assert sorted(p.name for p in session_dir.iterdir()) == ["session.json"]
    assert sorted(p.name for p in (tmp / "captures").iterdir()) == ["arland"]
    # a shard directory that is a symlink, even to a harmless place, is refused: shards are replaced by renames
    harmless = datasets / "harmless"
    harmless.mkdir()
    shard_dir = datasets / "out_d" / GOLDEN_DIR.name
    shard_dir.mkdir(parents=True)
    (shard_dir / "features_v1").symlink_to(harmless)
    with pytest.raises(build.BuildError, match="symlink"):
        build.write_shard(session, datasets / "out_d", spec=None, overwrite=True, dataset_roots=(datasets,))
    assert list(harmless.iterdir()) == []


def test_a_session_without_a_vehicle_card_is_refused(spec: FeatureSpec, tmp: Path) -> None:
    session = build.open_dc2_session(GOLDEN_DIR)  # no tools/ident/cards/1.8.0.13/S105_synthetic.json
    with pytest.raises(build.BuildError, match="vehicle card"):
        build.write_shard(session, tmp, spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(tmp,))


def test_eval_only_needs_its_flag(golden_session: Any, spec: FeatureSpec, tmp: Path) -> None:
    shard = _shard(
        build.write_shard(
            golden_session,
            tmp,
            spec=spec,
            routes_dir=ROUTES_DIR,
            include_eval_only=True,
            include_privileged=True,
            dataset_roots=(tmp,),
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
            "--dataset-root",
            str(tmp),
        ]
    )
    assert code == 0
    assert Shard(tmp / GOLDEN_DIR.name / "features_v1").n_steps == 68
    elsewhere = tmp.parent / (tmp.name + "_elsewhere")
    with pytest.raises(build.BuildError, match="not under a dataset root"):
        build.main(["--session", str(GOLDEN_DIR), "--out", str(elsewhere), "--dataset-root", str(tmp)])
    assert not elsewhere.exists()
