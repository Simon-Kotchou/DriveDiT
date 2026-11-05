"""Closed-form ridge, R² and segment-respecting folds (drivedit.eval.linear) on data with known answers."""

from __future__ import annotations

import numpy as np
import pytest
from drivedit.eval.linear import (
    check_segments,
    cross_val_predict,
    fit_ridge,
    r2_score,
    segment_folds,
    segment_rows,
)
from drivedit.eval.strata import SPEED_BIN_NAMES, speed_bin


def test_ridge_recovers_a_noiseless_linear_map() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)) * [1.0, 10.0, 0.1, 3.0]
    coef = np.array([[1.5, -2.0], [0.2, 0.0], [-7.0, 1.0], [0.0, 0.5]])
    y = x @ coef + [3.0, -1.0]
    fit = fit_ridge(x, y, lam=1e-9)
    np.testing.assert_allclose(fit.coef, coef, atol=1e-6)
    np.testing.assert_allclose(fit.intercept, [3.0, -1.0], atol=1e-6)


def test_wide_features_use_the_dual_and_match_the_primal() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 80))
    y = rng.normal(size=(30, 2))
    wide = fit_ridge(x, y, lam=2.0)
    # the same estimator in the primal form, solved directly
    z = (x - x.mean(0)) / x.std(0)
    beta = np.linalg.solve(z.T @ z + 2.0 * np.eye(80), z.T @ (y - y.mean(0)))
    np.testing.assert_allclose(wide.coef, beta / x.std(0)[:, None], atol=1e-8)


def test_constant_columns_get_no_weight_and_clip_limits_predictions_to_the_training_range() -> None:
    x = np.stack([np.linspace(0.0, 1.0, 50), np.full(50, 7.0)], axis=1)
    fit = fit_ridge(x, 2.0 * x[:, 0], lam=1e-9)
    assert list(fit.live) == [True, False]
    assert fit.coef[1, 0] == 0.0
    np.testing.assert_allclose(fit.predict([[100.0, 7.0]]), [[200.0]], atol=1e-6)  # linear extrapolation
    clipped = fit_ridge(x, 2.0 * x[:, 0], lam=1e-9, clip=True)
    # 100 lies far outside the fitted [0, 1]: it is clipped to 1, so the prediction is 2, not 200
    np.testing.assert_allclose(clipped.predict([[100.0, 7.0]]), [[2.0]], atol=1e-6)


def test_r2_perfect_mean_and_constant() -> None:
    y = np.arange(10.0)
    assert r2_score(y, y)[0] == pytest.approx(1.0)
    assert r2_score(y, np.full(10, y.mean()))[0] == pytest.approx(0.0)
    assert np.isnan(r2_score(np.ones(10), np.arange(10.0))[0])


def test_blocked_folds_cover_every_row_once_and_purge_inside_segments() -> None:
    segments = [(0, 40), (40, 70), (100, 160)]
    folds = segment_folds(segments, n_folds=4, gap=3)
    assert folds.kind == "blocked"
    tested = np.sort(np.concatenate([f.test for f in folds.folds]))
    np.testing.assert_array_equal(tested, segment_rows(segments))
    for fold in folds.folds:
        assert not set(fold.train) & set(fold.test)
        for start, stop in segments:
            test_rows = fold.test[(fold.test >= start) & (fold.test < stop)]
            if test_rows.size:
                train_rows = fold.train[(fold.train >= start) & (fold.train < stop)]
                near = (train_rows >= test_rows.min() - 3) & (train_rows <= test_rows.max() + 3)
                assert not near.any()


def test_purge_does_not_reach_across_a_segment_boundary() -> None:
    folds = segment_folds([(0, 10), (10, 20)], n_folds=2, gap=5)
    first = folds.folds[0]  # tests rows 0..9: row 10 starts the other segment and stays in training
    np.testing.assert_array_equal(first.test, np.arange(10))
    assert 10 in first.train


def test_two_sessions_give_held_out_session_folds() -> None:
    folds = segment_folds([(0, 10), (10, 20), (20, 35)], n_folds=5, gap=2, groups=["a", "a", "b"])
    assert folds.kind == "group"
    assert len(folds.folds) == 2
    np.testing.assert_array_equal(folds.folds[1].test, np.arange(20, 35))
    np.testing.assert_array_equal(folds.folds[1].train, np.arange(0, 20))


def test_one_session_falls_back_to_blocked_folds() -> None:
    assert segment_folds([(0, 30)], n_folds=3, gap=0, groups=["only"]).kind == "blocked"


def test_bad_segments_and_too_many_folds_raise() -> None:
    with pytest.raises(ValueError, match="overlap"):
        check_segments([(0, 10), (5, 12)], 20)
    with pytest.raises(ValueError, match="inside"):
        check_segments([(0, 21)], 20)
    with pytest.raises(ValueError, match="blocked folds"):
        segment_folds([(0, 3)], n_folds=5, gap=0)


def test_cross_val_predict_scores_a_signal_and_not_noise() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, 3))
    folds = segment_folds([(0, 300)], n_folds=5, gap=0)
    signal = x @ [1.0, -1.0, 0.5] + 0.1 * rng.normal(size=300)
    noise = rng.normal(size=300)
    assert r2_score(signal, cross_val_predict(x, signal[:, None], folds, lam=1.0))[0] > 0.95
    assert r2_score(noise, cross_val_predict(x, noise[:, None], folds, lam=1.0))[0] < 0.05


def test_speed_bins_follow_the_capture_program_edges() -> None:
    kmh = np.array([0.0, 4.9, 5.0, 19.9, 20.0, 34.9, 35.0, 49.9, 50.0, 120.0, -40.0])
    names = [SPEED_BIN_NAMES[b] for b in speed_bin(kmh / 3.6)]
    assert names == ["0-5", "0-5", "5-20", "5-20", "20-35", "20-35", "35-50", "35-50", ">50", ">50", "35-50"]
