"""State probes and the explained action information (drivedit.eval.probes) on data with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from drivedit.eval.probes import explained_action_information, state_probe_r2

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


def test_eai_positive_when_the_action_helps() -> None:
    rng = np.random.default_rng(6)
    loss_true = rng.uniform(0.5, 1.0, size=600)
    loss_null = loss_true + 0.1 + 0.02 * rng.normal(size=600)
    bins = np.repeat(["5-20", "20-35", "35-50"], 200)
    eai = explained_action_information(loss_null, loss_true, buckets=list(bins))
    assert eai.eai == pytest.approx(0.1, abs=0.01)
    assert eai.lower_95 > 0.0
    assert eai.positive_in_every_bucket
    assert [b.bucket for b in eai.buckets] == ["5-20", "20-35", "35-50"]


def test_eai_zero_when_the_action_is_ignored_or_shuffled() -> None:
    rng = np.random.default_rng(7)
    loss = rng.uniform(0.5, 1.0, size=400)
    ignored = explained_action_information(loss, loss)
    assert ignored.eai == 0.0
    assert not ignored.positive_in_every_bucket
    shuffled = explained_action_information(loss[rng.permutation(400)], loss)
    assert shuffled.eai == pytest.approx(0.0, abs=1e-12)  # a permutation keeps the mean
    assert shuffled.lower_95 < 0.0
    assert not shuffled.positive_in_every_bucket


def test_eai_fails_the_bar_when_one_bucket_ignores_the_action() -> None:
    loss_true = np.full(300, 1.0)
    loss_null = np.concatenate([np.full(200, 1.2), np.full(100, 1.0)]) + np.tile([0.01, -0.01], 150)
    buckets = ["20-35"] * 200 + ["35-50"] * 100
    eai = explained_action_information(loss_null, loss_true, buckets=buckets)
    assert eai.eai > 0.0
    assert not eai.positive_in_every_bucket


def test_eai_standard_error_counts_clips_not_steps() -> None:
    rng = np.random.default_rng(8)
    clip_effect = np.repeat(rng.normal(0.05, 0.1, size=20), 50)  # 20 clips of 50 steps, correlated within a clip
    diff = clip_effect + 0.01 * rng.normal(size=1000)
    naive = explained_action_information(diff, np.zeros(1000))
    clustered = explained_action_information(diff, np.zeros(1000), clusters=list(np.repeat(np.arange(20), 50)))
    assert clustered.n_clusters == 20
    assert clustered.se > 4.0 * naive.se
