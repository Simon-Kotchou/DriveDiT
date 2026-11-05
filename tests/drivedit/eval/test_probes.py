"""State probes and the explained action information (drivedit.eval.probes) on data with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from drivedit.eval.probes import explained_action_information, state_probe_r2, t_critical

SEGMENTS = [(0, 300), (300, 520), (520, 800)]


def _state(seed: int, n: int = 800) -> np.ndarray:
    """A slowly varying [speed, yaw rate] trajectory."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(size=(n, 2)), axis=0) * [0.1, 0.01] + [10.0, 0.0]


def test_probe_reads_a_linearly_encoded_state() -> None:
    rng = np.random.default_rng(0)
    state = _state(1)
    features = state @ rng.normal(size=(2, 32)) + 0.001 * rng.normal(size=(800, 32))
    probe = state_probe_r2(features, state, segments=SEGMENTS, names=["speed_mps", "yaw_rate_rps"], gap=2)
    assert probe.r2_of("speed_mps") > 0.99
    assert probe.r2_of("yaw_rate_rps") > 0.99
    assert probe.cv == "blocked"
    assert len(probe.fold_r2) == 5


def test_probe_on_unrelated_or_shuffled_features_reads_nothing() -> None:
    rng = np.random.default_rng(2)
    state = _state(3)
    unrelated = rng.normal(size=(800, 16))
    assert max(state_probe_r2(unrelated, state, segments=SEGMENTS).r2) < 0.05
    encoded = state @ rng.normal(size=(2, 16))
    shuffled = encoded[rng.permutation(800)]
    assert max(state_probe_r2(shuffled, state, segments=SEGMENTS).r2) < 0.05


def test_wide_probe_features_and_held_out_sessions() -> None:
    rng = np.random.default_rng(4)
    state = _state(5)
    wide = state @ rng.normal(size=(2, 2000))  # D > N: solved in the dual
    probe = state_probe_r2(wide, state, segments=SEGMENTS, groups=["s1", "s1", "s2"], lam=1e-3)
    assert probe.cv == "group"
    assert len(probe.fold_r2) == 2
    assert min(probe.r2) > 0.95


def test_probe_input_checks() -> None:
    with pytest.raises(ValueError, match="names"):
        state_probe_r2(np.zeros((10, 2)), np.zeros((10, 2)), segments=[(0, 10)], names=["one"])
    with pytest.raises(ValueError, match="rows"):
        state_probe_r2(np.zeros((10, 2)), np.zeros((9, 2)), segments=[(0, 9)])


def test_eai_passes_when_the_action_helps_in_every_bucket() -> None:
    rng = np.random.default_rng(6)
    loss_true = rng.uniform(0.5, 1.0, size=600)
    loss_null = loss_true + 0.1 + 0.02 * rng.normal(size=600)
    bins = np.repeat(["5-20", "20-35", "35-50"], 200)
    eai = explained_action_information(loss_null, loss_true, buckets=list(bins))
    assert eai.value == pytest.approx(0.1, abs=0.01)
    assert eai.overall.lower_95 > 0.0
    assert eai.gate == "pass"
    assert eai.reasons == ()
    assert [b.bucket for b in eai.buckets] == ["5-20", "20-35", "35-50"]
    assert all(b.status == "pass" for b in eai.buckets)


def test_eai_fails_when_the_action_is_ignored_or_shuffled() -> None:
    rng = np.random.default_rng(7)
    loss = rng.uniform(0.5, 1.0, size=400)
    ignored = explained_action_information(loss, loss)
    assert ignored.value == 0.0
    assert ignored.gate == "fail"
    shuffled = explained_action_information(loss[rng.permutation(400)], loss)
    assert shuffled.value == pytest.approx(0.0, abs=1e-12)  # a permutation keeps the mean
    assert shuffled.overall.lower_95 < 0.0
    assert shuffled.gate == "fail"


def test_eai_fails_the_bar_when_one_bucket_ignores_the_action() -> None:
    loss_true = np.full(300, 1.0)
    loss_null = np.concatenate([np.full(200, 1.2), np.full(100, 1.0)]) + np.tile([0.01, -0.01], 150)
    buckets = ["20-35"] * 200 + ["35-50"] * 100
    eai = explained_action_information(loss_null, loss_true, buckets=buckets)
    assert eai.value > 0.0
    assert eai.gate == "fail"
    assert [b.status for b in eai.buckets] == ["pass", "fail"]
    assert any(r.startswith("35-50") for r in eai.reasons)


def test_an_under_sampled_bucket_blocks_a_pass_and_never_disappears() -> None:
    """Review round 1: clips [1, 1, -10] in buckets [common, common, rare] once passed with an overall EAI of -2.667,
    because the one-clip bucket was dropped from the gate."""
    args = ([1.0, 1.0, -10.0], [0.0, 0.0, 0.0])
    kwargs = {"buckets": ["common", "common", "rare"], "clusters": [0, 1, 2]}
    default = explained_action_information(*args, **kwargs)
    assert default.value == pytest.approx(-8.0 / 3.0)
    assert default.gate == "insufficient"  # 3 clips against the default 50
    assert [b.bucket for b in default.buckets] == ["common", "rare"]
    small = explained_action_information(*args, **kwargs, min_clusters=2)
    assert small.gate == "fail"  # the overall EAI is negative
    assert {b.bucket: b.status for b in small.buckets} == {"common": "pass", "rare": "insufficient"}
    # positive everywhere it was sampled, but the rare bucket has one clip: still not a pass
    lucky = explained_action_information([1.0, 1.01, 1.02], [0.0, 0.0, 0.0], **kwargs, min_clusters=2)
    assert lucky.overall.status == "pass"
    assert lucky.gate == "insufficient"
    assert lucky.reasons == ("rare: 1 clip(s), fewer than the 2 required",)


def test_a_required_bucket_without_rows_is_insufficient() -> None:
    rng = np.random.default_rng(9)
    loss_true = rng.uniform(0.5, 1.0, size=200)
    eai = explained_action_information(
        loss_true + 0.2, loss_true, buckets=["35-50"] * 200, required_buckets=["35-50", ">50"]
    )
    assert [b.bucket for b in eai.buckets] == ["35-50", ">50"]
    assert eai.buckets[1].n == 0
    assert eai.buckets[1].status == "insufficient"
    assert eai.gate == "insufficient"


def test_eai_bound_uses_student_t_over_clips() -> None:
    """Review round 1: two clips [0.1, 0.3] gave a normal bound of 0.004 (a pass); with t at 1 df it is -1.071."""
    eai = explained_action_information([0.1, 0.3], [0.0, 0.0], clusters=[0, 1], min_clusters=2)
    assert eai.overall.se == pytest.approx(0.1)
    assert eai.overall.t_crit == pytest.approx(12.7062, abs=1e-4)
    assert eai.overall.lower_95 == pytest.approx(-1.0706, abs=1e-3)
    assert eai.gate == "fail"


@pytest.mark.parametrize(
    ("df", "expected"),
    [(1, 12.7062), (2, 4.3027), (3, 3.1824), (5, 2.5706), (10, 2.2281), (30, 2.0423), (49, 2.0096), (120, 1.9799)],
)
def test_t_critical_matches_the_t_table(df: int, expected: float) -> None:
    assert t_critical(df) == pytest.approx(expected, abs=1e-4)


def test_eai_standard_error_counts_clips_not_steps() -> None:
    rng = np.random.default_rng(8)
    clip_effect = np.repeat(rng.normal(0.05, 0.1, size=20), 50)  # 20 clips of 50 steps, correlated within a clip
    diff = clip_effect + 0.01 * rng.normal(size=1000)
    naive = explained_action_information(diff, np.zeros(1000))
    clustered = explained_action_information(diff, np.zeros(1000), clusters=list(np.repeat(np.arange(20), 50)))
    assert clustered.overall.n_clusters == 20
    assert clustered.overall.se > 4.0 * naive.overall.se
    assert clustered.gate == "insufficient"  # 20 clips: fewer than the 50 a gate needs


def test_eai_rejects_a_gate_without_a_standard_error() -> None:
    with pytest.raises(ValueError, match="min_clusters"):
        explained_action_information([1.0, 2.0], [0.0, 0.0], min_clusters=1)
