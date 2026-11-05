"""E0 (drivedit.eval.e0): synthetic transitions with known answers, the golden dc-2.0 fixture, and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drivedit.eval.e0 import E0Report, SessionTransitions, e0_action_gain, main, session_transitions

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "capture_dc2"
GOLDEN = FIXTURES / "golden" / "arland_T0b_20261025T091512Z_wb_i00_5e00"
CARD = FIXTURES / "cards" / "S105_synthetic.json"
STEP_S = 0.2  # features_v1 offsets: decision_hz 5


def _drive(
    seed: int,
    *,
    n_segments: int = 3,
    length: int = 400,
    responds: bool = True,
    constant_speed: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Toy 5 Hz kinematics with commands independent of the state: yaw[t+1] = v[t] x kappa[t] (an instant llc),
    v[t+1] = v[t] + a[t] x 0.2 s. With ``responds`` False the next state ignores the commands; with
    ``constant_speed`` the car holds 11.1 m/s and a_target is 0 throughout (the first T0L session's cruise)."""
    rng = np.random.default_rng(seed)
    state, action, nxt, segments = [], [], [], []
    for _ in range(n_segments):
        v, yaw, acc = 11.1 if constant_speed else rng.uniform(4.0, 12.0), 0.0, 0.0
        start = len(state)
        for _ in range(length):
            kappa = rng.normal(0.0, 0.02)
            a = 0.0 if constant_speed else rng.uniform(-1.5, 1.5) - 0.2 * (v - 8.0)
            state.append([v, yaw, acc, 1.0])
            action.append([kappa, a, 1.0])
            if responds:
                yaw = v * kappa + rng.normal(0.0, 0.002)
                dv = a * STEP_S + rng.normal(0.0, 0.003 if constant_speed else 0.02)
            else:
                yaw = 0.9 * yaw + rng.normal(0.0, 0.05)
                dv = rng.normal(0.0, 0.2) - 0.05 * (v - 8.0)
            v, acc = v + dv, dv / STEP_S
            nxt.append([v, yaw, acc, 1.0])
        segments.append((start, len(state)))
    return np.asarray(state), np.asarray(action), np.asarray(nxt), segments


def _e0(state: np.ndarray, action: np.ndarray, nxt: np.ndarray, segments: list[tuple[int, int]]) -> E0Report:
    return e0_action_gain(state, action, nxt, segments=segments, history=2)


def test_commands_that_drive_the_next_state_show_a_gain() -> None:
    report = _e0(*_drive(0))
    # known answers: d_yaw = v*kappa - yaw[t], so the state explains the half that is -yaw[t] and the command the
    # other half; d_speed is almost all a_target (the state predicts only its small mean-reverting part)
    yaw, speed = report.target("d_yaw_rate_rps"), report.target("d_speed_mps")
    assert yaw.r2_state == pytest.approx(0.5, abs=0.05)
    assert yaw.gain == pytest.approx(0.5, abs=0.05)
    assert speed.r2_state < 0.1
    assert speed.gain > 0.85
    for result in (yaw, speed):
        assert result.can_show
        assert result.shown, result
        assert result.reasons == ()
        assert result.r2_state_action > 0.98
        assert all(g > 0.0 for g in result.fold_gains)
    assert report.cv == "blocked"
    assert report.gap == 3


def test_commands_the_next_state_ignores_show_none() -> None:
    report = _e0(*_drive(1, responds=False))
    for result in report.targets:
        assert result.can_show  # the commands vary; they just do not matter
        assert not result.shown
        assert abs(result.gain) < 0.02
        assert any("folds" in r or "pooled" in r for r in result.reasons)  # the reasons say which way it lost
        assert any("out-of-fold R² from the state features" in r for r in result.reasons)
        assert not any("predictable" in r for r in result.reasons)  # R² <= 0 here: no claim the state predicts it


def test_shuffled_commands_show_no_gain() -> None:
    state, action, nxt, segments = _drive(2)
    shuffled = action[np.random.default_rng(3).permutation(len(action))]
    report = _e0(state, shuffled, nxt, segments)
    for result in report.targets:
        assert not result.shown
        assert result.gain < 0.02


def test_constant_speed_cannot_show_a_speed_gain() -> None:
    report = _e0(*_drive(4, constant_speed=True))
    speed = report.target("d_speed_mps")
    assert not speed.can_show
    assert not speed.shown
    assert "a_target" in speed.reasons[0]
    assert all(a.constant for a in speed.actions)
    yaw = report.target("d_yaw_rate_rps")
    assert yaw.can_show
    assert yaw.shown
    bins = {b.speed_bin: b.n for b in yaw.speed_bins}
    assert bins["35-50"] == report.n_rows  # 11.1 m/s = 40 km/h


def test_rows_outside_segments_and_before_the_history_are_not_used() -> None:
    state, action, nxt, segments = _drive(5, n_segments=2, length=120)
    inner = [(s + 10, e - 10) for s, e in segments]
    base = e0_action_gain(state, action, nxt, segments=inner, history=3)
    assert base.n_rows == sum(e - s - 3 for s, e in inner)
    noisy = state.copy()
    noisy[: inner[0][0]] = 1e6  # rows before the first segment
    noisy[inner[0][1] : inner[1][0]] = -1e6  # between the segments, including the history the next one may not use
    other = e0_action_gain(noisy, action, nxt, segments=inner, history=3)
    assert other.to_json() == base.to_json()


def test_bad_inputs_raise() -> None:
    state, action, nxt, segments = _drive(6, n_segments=1, length=50)
    with pytest.raises(ValueError, match=r"\[N, 3\]"):
        e0_action_gain(state, action[:, :2], nxt, segments=segments)
    with pytest.raises(ValueError, match="no segment is longer"):
        e0_action_gain(state, action, nxt, segments=[(0, 3)], history=5)


@pytest.fixture(scope="module")
def golden_rows() -> SessionTransitions:
    return session_transitions(GOLDEN, vehicle_card=CARD)


def _golden_decisions() -> tuple[list[dict[str, object]], list[dict[str, int]]]:
    golden = json.loads((FIXTURES / "golden.json").read_text())
    return golden["decision_steps"], golden["segments"]


def test_golden_session_rows_follow_the_contract_offsets(golden_rows: SessionTransitions) -> None:
    """Row t holds state[t] and action_cmd[t] of one decision step and next_state = state[t+1] of the next, inside the
    reader's segments (the expected values come from golden.json, written by the fixture's own simulation)."""
    rows = golden_rows
    decisions, segments = _golden_decisions()
    index = {d["decision_step"]: i for i, d in enumerate(decisions)}
    expected_state, expected_action, expected_next = [], [], []
    for seg in segments:
        first, last = index[seg["first_decision_step"]], index[seg["last_decision_step"]]
        for t in range(first, last):
            expected_state.append(decisions[t]["state"])
            expected_action.append(decisions[t]["action_cmd"])
            expected_next.append(decisions[t + 1]["state"])
    assert rows.segments == ((0, 25), (25, 50))
    np.testing.assert_allclose(rows.state, np.asarray(expected_state, dtype=float), atol=1e-4)
    np.testing.assert_allclose(rows.action_cmd, np.asarray(expected_action, dtype=float), atol=1e-4)
    np.testing.assert_allclose(rows.next_state, np.asarray(expected_next, dtype=float), atol=1e-4)


def test_golden_session_e0_runs_and_can_show_both_gains(golden_rows: SessionTransitions) -> None:
    rows = golden_rows
    report = e0_action_gain(rows.state, rows.action_cmd, rows.next_state, segments=rows.segments, history=1)
    assert report.n_rows == 48
    for result in report.targets:
        assert result.can_show  # the golden drive steers and accelerates
        assert np.isfinite(result.r2_state)
        assert np.isfinite(result.r2_state_action)
        assert len(result.fold_gains) == 5


def test_cli_prints_the_report_as_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--session", str(GOLDEN), "--vehicle-card", str(CARD), "--history", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["sessions"] == [GOLDEN.name]
    assert out["history"] == 1
    assert [t["target"] for t in out["targets"]] == ["d_yaw_rate_rps", "d_speed_mps"]
