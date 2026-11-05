"""dd-train's data guard: only Arland ``train`` sessions at the model's decision rate train or enter normalisation
statistics, and Everon, ``heldout/`` and ``SEALED`` data are refused before anything under them is read."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from drivedit.train import DataConfig, RunConfig
from drivedit.train.data import (
    SEALED_MAP_IDS,
    SEALED_PATH_PARTS,
    TRAINING_SPLITS,
    SealedDataError,
    TrainingDataError,
    check_training_sessions,
    open_training_windows,
)
from tests.drivedit.model.toy import tiny_config

NAV = importlib.import_module("drivedit_nav.nav_v1")  # the route schema: MAP_SPLITS, the sealed rules


def _session(root: Path, session: str, **meta: object) -> str:
    path = root / session / "features_v1"
    path.mkdir(parents=True)
    fields = {"schema": "drivedit.features_v1-shard/1", "map_id": "arland", "split": "train", "decision_hz": 5.0}
    (path / "meta.json").write_text(json.dumps({**fields, **meta}))
    return session


def _data(root: Path, *sessions: str) -> DataConfig:
    return DataConfig(root=str(root), sessions=sessions, codec_id="cadre-1x16x16")


def test_arland_train_sessions_pass(tmp_path: Path) -> None:
    a, b = _session(tmp_path, "arland_a"), _session(tmp_path, "arland_b")
    assert check_training_sessions(_data(tmp_path, a, b), 5.0) == [tmp_path / s / "features_v1" for s in (a, b)]


ROUTE_SPLITS = [(m, s) for m, splits in NAV.MAP_SPLITS.items() for s in splits]


@pytest.mark.parametrize(("map_id", "split"), [*ROUTE_SPLITS, ("arland", "val_c"), ("arland", None)])
def test_every_other_split_is_refused(tmp_path: Path, map_id: str, split: str | None) -> None:
    """Every route split tag of every map (drivedit_nav MAP_SPLITS), val_c and a shard without a route split."""
    session = _session(tmp_path, "s", map_id=map_id, split=split)
    if (map_id, split) == ("arland", "train"):
        check_training_sessions(_data(tmp_path, session), 5.0)
        return
    sealed = map_id in NAV.SEALED_MAP_IDS or str(split).startswith("everon")
    with pytest.raises(SealedDataError if sealed else TrainingDataError, match="sealed" if sealed else "never trains"):
        check_training_sessions(_data(tmp_path, session), 5.0)


def test_the_rules_match_the_route_schema_and_the_reader() -> None:
    assert set(TRAINING_SPLITS) <= set(NAV.MAP_SPLITS)
    assert all(set(splits) <= set(NAV.MAP_SPLITS[m]) for m, splits in TRAINING_SPLITS.items())
    assert tuple(SEALED_PATH_PARTS) == tuple(NAV.SEALED_PATH_PARTS)
    assert tuple(SEALED_MAP_IDS) == tuple(NAV.SEALED_MAP_IDS)


@pytest.mark.parametrize(
    ("root_part", "session"),
    [("heldout/everon", "arland_x"), ("SEALED", "arland_x"), ("runs", "everon_T4E_x"), ("runs", "EVERON_x")],
)
def test_sealed_paths_are_refused_before_any_read(tmp_path: Path, root_part: str, session: str) -> None:
    """Nothing exists under these paths: the refusal comes before meta.json is opened."""
    with pytest.raises(SealedDataError, match="never reads"):
        check_training_sessions(_data(tmp_path / root_part, session), 5.0)


def test_a_link_into_a_sealed_directory_is_refused(tmp_path: Path) -> None:
    sealed = tmp_path / "heldout" / "everon" / "SEALED"
    _session(sealed, "renamed")
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "arland_copy").symlink_to(sealed / "renamed")
    with pytest.raises(SealedDataError):
        check_training_sessions(_data(tmp_path / "datasets", "arland_copy"), 5.0)


def test_the_decision_rate_must_match_the_model(tmp_path: Path) -> None:
    session = _session(tmp_path, "arland_10hz", decision_hz=10.0)
    with pytest.raises(TrainingDataError, match=r"decision_hz 10\.0"):
        check_training_sessions(_data(tmp_path, session), 5.0)
    assert check_training_sessions(_data(tmp_path, session), 10.0)


def test_open_training_windows_checks_before_the_data_layer(tmp_path: Path) -> None:
    """On this base DD-11 is absent, so reaching the data layer would raise RuntimeError instead."""
    good, bad = _session(tmp_path, "arland_train"), _session(tmp_path, "arland_val", split="val_l")
    cfg = RunConfig(run_id="guard", model=tiny_config(), data=_data(tmp_path, good, bad))
    with pytest.raises(TrainingDataError, match="val_l"):
        open_training_windows(cfg)
