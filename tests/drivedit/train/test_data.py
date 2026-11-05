"""dd-train's data guard: only Arland ``train`` sessions outside VAL-C, at the model's decision rate, train or enter
normalisation statistics; Everon, ``heldout/`` and ``SEALED`` data are refused before anything is read, including
through a symlinked meta.json, array or latents resource."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
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
    is_val_c,
    open_training_windows,
)
from tests.drivedit.model.toy import tiny_config

NAV = importlib.import_module("drivedit_nav.nav_v1")  # the route schema: MAP_SPLITS, the sealed rules


CODEC = "cadre-1x16x16"
#: Codex's DD-20 round-2 probe: a train-route session that dc-2.0 section 3.2 puts in VAL-C (hash mod 10 == 0).
VAL_C_SESSION = "arland_T0b_20261025T091512Z_wb_i00_0004"


def _session(root: Path, session: str, **meta: object) -> str:
    """A shard folder with only meta.json, and a latents folder with a stand-in latents.npy (never read here)."""
    path = root / session / "features_v1"
    path.mkdir(parents=True)
    fields = {"schema": "drivedit.features_v1-shard/1", "session_id": session, "map_id": "arland", "split": "train"}
    (path / "meta.json").write_text(json.dumps({**fields, "decision_hz": 5.0, **meta}))
    latents = root / session / "latents" / CODEC
    latents.mkdir(parents=True)
    (latents / "latents.npy").write_bytes(b"not read by the guard")
    return session


def _data(root: Path, *sessions: str) -> DataConfig:
    return DataConfig(root=str(root), sessions=sessions, codec_id=CODEC)


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
    session = _session(tmp_path, "arland_rate10", decision_hz=10.0)
    with pytest.raises(TrainingDataError, match=r"decision_hz 10\.0"):
        check_training_sessions(_data(tmp_path, session), 5.0)
    assert check_training_sessions(_data(tmp_path, session), 10.0)


def test_open_training_windows_checks_before_the_data_layer(tmp_path: Path) -> None:
    """On this base DD-11 is absent, so reaching the data layer would raise RuntimeError instead."""
    good, bad = _session(tmp_path, "arland_train"), _session(tmp_path, "arland_val", split="val_l")
    cfg = RunConfig(run_id="guard", model=tiny_config(), data=_data(tmp_path, good, bad))
    with pytest.raises(TrainingDataError, match="val_l"):
        open_training_windows(cfg)


def test_hash_selected_val_c_sessions_never_train(tmp_path: Path) -> None:
    """Meta says train, but dc-2.0 section 3.2 (and the validator's C02) makes the session VAL-C."""
    assert int(hashlib.sha256(VAL_C_SESSION.encode("ascii")).hexdigest()[:8], 16) % 10 == 0
    assert is_val_c(VAL_C_SESSION)
    assert not is_val_c("arland_a")
    session = _session(tmp_path, VAL_C_SESSION)
    with pytest.raises(TrainingDataError, match=r"disagrees with dc-2\.0 section 3\.2"):
        check_training_sessions(_data(tmp_path, session), 5.0)


def test_meta_must_name_its_own_session(tmp_path: Path) -> None:
    session = _session(tmp_path, "arland_renamed_dir", session_id="arland_a")
    with pytest.raises(TrainingDataError, match="names session 'arland_a'"):
        check_training_sessions(_data(tmp_path, session), 5.0)


@pytest.mark.parametrize("map_id", ["Everon", "EVERON"])
def test_mixed_case_everon_is_sealed(tmp_path: Path, map_id: str) -> None:
    session = _session(tmp_path, "arland_a", map_id=map_id)
    with pytest.raises(SealedDataError, match="sealed test map"):
        check_training_sessions(_data(tmp_path, session), 5.0)


def _sealed_copy(tmp_path: Path, sealed_part: str) -> Path:
    """A permitted-looking Arland train session stored under a sealed directory (its meta would pass if read)."""
    root = tmp_path / sealed_part / "everon"
    _session(root, "arland_a")
    (root / "arland_a" / "features_v1" / "state.npy").write_bytes(b"sealed")
    (root / "arland_a" / "latents" / CODEC / "valid.npy").write_bytes(b"sealed")
    return root / "arland_a"


def _link_meta(session_dir: Path, sealed: Path) -> None:
    (session_dir / "features_v1" / "meta.json").unlink()
    (session_dir / "features_v1" / "meta.json").symlink_to(sealed / "features_v1" / "meta.json")


def _link_array(session_dir: Path, sealed: Path) -> None:
    (session_dir / "features_v1" / "state.npy").symlink_to(sealed / "features_v1" / "state.npy")


def _link_latents_dir(session_dir: Path, sealed: Path) -> None:
    latents = session_dir / "latents" / CODEC
    (latents / "latents.npy").unlink()
    latents.rmdir()
    latents.symlink_to(sealed / "latents" / CODEC)


def _link_latents_file(session_dir: Path, sealed: Path) -> None:
    (session_dir / "latents" / CODEC / "valid.npy").symlink_to(sealed / "latents" / CODEC / "valid.npy")


@pytest.mark.parametrize("sealed_part", ["SEALED", "heldout"])
@pytest.mark.parametrize("link", [_link_meta, _link_array, _link_latents_dir, _link_latents_file])
def test_a_resource_linked_into_sealed_data_is_refused(
    tmp_path: Path, sealed_part: str, link: Callable[[Path, Path], None]
) -> None:
    """Codex's round-2 probes: the shard folder is permitted, but meta.json, an array, the latents folder or a latents
    file resolves into sealed data. Each is refused before meta.json is parsed."""
    sealed = _sealed_copy(tmp_path, sealed_part)
    session = _session(tmp_path / "datasets", "arland_ok")
    assert check_training_sessions(_data(tmp_path / "datasets", session), 5.0)  # permitted before the link
    link(tmp_path / "datasets" / session, sealed)
    with pytest.raises(SealedDataError, match="never reads"):
        check_training_sessions(_data(tmp_path / "datasets", session), 5.0)
