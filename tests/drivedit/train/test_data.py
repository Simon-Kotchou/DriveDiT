"""dd-train's data guard: only Arland ``train`` sessions outside VAL-C, at the model's decision rate, train or enter
normalisation statistics; Everon, ``heldout/`` and ``SEALED`` data are refused before anything is read, including
through a symlinked meta.json, array or latents resource, a redirected shard folder, or a file named in meta.json.

The bypass tests run ``open_training_windows`` on a stand-in for DD-11's reader as it was before its link hardening
(31f328e): it resolves the shard folder, opens the arrays named in its meta.json there, and finds the latents beside
the resolved folder, with no integrity checks of its own. It records every path it opens."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import types
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from torch.utils.data import Dataset

from drivedit.train import DataConfig, RunConfig, data
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
T, LATENT = 4, (2, 1, 3)  # steps per stand-in session, and the latent shape C, h, w
#: Codex's DD-20 round-2 probe: a train-route session that dc-2.0 section 3.2 puts in VAL-C (hash mod 10 == 0).
VAL_C_SESSION = "arland_T0b_20261025T091512Z_wb_i00_0004"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _session(root: Path, session: str, *, arrays: dict[str, Any] | None = None, **meta: object) -> str:
    """A shard folder with meta.json and slot_id.npy, and a latents folder as DD-19's pre-encoder writes it: stand-in
    arrays (never read by the guard) and a meta.json naming this shard's session_id and slot_id.npy sha256."""
    path = root / session / "features_v1"
    path.mkdir(parents=True)
    np.save(path / "slot_id.npy", np.arange(T))
    slots = {"slot_id": {"file": "slot_id.npy", "sha256": _sha256(path / "slot_id.npy")}}
    fields = {"schema": "drivedit.features_v1-shard/1", "session_id": session, "map_id": "arland", "split": "train"}
    shard_meta = {**fields, "decision_hz": 5.0, "arrays": {**slots, **(arrays or {})}, **meta}
    (path / "meta.json").write_text(json.dumps(shard_meta))
    latents = root / session / "latents" / CODEC
    latents.mkdir(parents=True)
    for name in ("latents.npy", "valid.npy"):
        (latents / name).write_bytes(b"not read by the guard")
    latents_meta = {
        "schema": "drivedit.latents/1",
        "codec_id": CODEC,
        "session_id": shard_meta["session_id"],
        "aligned_to": {"file": "slot_id.npy", "sha256": slots["slot_id"]["sha256"]},
        "sha256": {"latents.npy": "x", "valid.npy": "y"},
    }
    (latents / "meta.json").write_text(json.dumps(latents_meta))
    return session


def _latents_meta(session_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((session_dir / "latents" / CODEC / "meta.json").read_text())
    return meta


def _write_latents_meta(session_dir: Path, meta: dict[str, Any]) -> None:
    (session_dir / "latents" / CODEC / "meta.json").write_text(json.dumps(meta))


def _data(root: Path, *sessions: str) -> DataConfig:
    return DataConfig(root=str(root), sessions=sessions, codec_id=CODEC)


def test_arland_train_sessions_pass(tmp_path: Path) -> None:
    a, b = _session(tmp_path, "arland_a"), _session(tmp_path, "arland_b")
    want = [(tmp_path / s / "features_v1").resolve() for s in (a, b)]
    assert check_training_sessions(_data(tmp_path, a, b), 5.0) == want


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


def _empty(folder: Path) -> None:
    for entry in folder.iterdir():
        entry.unlink()
    folder.rmdir()


def _link_latents_dir(session_dir: Path, sealed: Path) -> None:
    latents = session_dir / "latents" / CODEC
    _empty(latents)
    latents.symlink_to(sealed / "latents" / CODEC)


def _link_latents_file(session_dir: Path, sealed: Path) -> None:
    (session_dir / "latents" / CODEC / "valid.npy").unlink()
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


# --------------------------------------------------------------------------------------------- the reader's paths


@dataclass(frozen=True)
class _Stats:
    mean: tuple[float, ...]
    std: tuple[float, ...]


class _Windows(Dataset[dict[str, Any]]):
    def __init__(self, index: Any, codec_id: str, *, stats: _Stats, channels: Any, aux: Any) -> None:
        self.index, self.stats = index, stats

    def __len__(self) -> int:
        return len(self.index.shards)

    def __getitem__(self, i: int) -> dict[str, Any]:
        raise NotImplementedError


def _pre_hardening_reader(opened: list[Path]) -> dict[str, types.ModuleType]:
    """The paths DD-11's Shard and Latents open at 31f328e, with every opened path recorded and no checks."""

    def load(path: Path) -> npt.NDArray[Any]:
        opened.append(path.resolve())
        return np.load(path, allow_pickle=False)

    class Shard:
        def __init__(self, path: Path) -> None:
            self.path = Path(path).resolve()
            opened.append(self.path / "meta.json")
            self.meta = json.loads((self.path / "meta.json").read_text())
            self.arrays = {n: load(self.path / str(info["file"])) for n, info in self.meta["arrays"].items()}

        def latents(self, codec_id: str) -> types.SimpleNamespace:
            folder = self.path.parent / "latents" / codec_id
            return types.SimpleNamespace(latents=load(folder / "latents.npy"), valid=load(folder / "valid.npy"))

    class WindowIndex:
        def __init__(self, shards: list[Shard], length: int, stride: int = 1) -> None:
            self.shards = shards

    shard = types.ModuleType("drivedit.data.shard")
    shard.Shard, shard.WindowIndex, shard.LatentStats = Shard, WindowIndex, _Stats
    torch_module = types.ModuleType("drivedit.data.torch")
    torch_module.LatentWindowDataset = _Windows
    return {"drivedit.data.shard": shard, "drivedit.data.torch": torch_module}


def _trainable(root: Path, session: str, *, value: float = 1.0) -> Path:
    """A complete session in the DD-11 / DD-19 layout: a state array named in meta.json, latents with their meta."""
    _session(root, session, arrays={"state": {"file": "state.npy"}})
    np.save(root / session / "features_v1" / "state.npy", np.zeros((T, 4), np.float32))
    latents = root / session / "latents" / CODEC
    (latents / "latents.npy").unlink()
    np.save(latents / "latents.npy", np.full((T, *LATENT), value, np.float16))
    np.save(latents / "valid.npy", np.ones(T, bool))
    return root / session


def _train(monkeypatch: pytest.MonkeyPatch, root: Path, session: str, opened: list[Path]) -> None:
    """open_training_windows on the stand-in reader, which appends every path it opens to ``opened``."""
    modules = _pre_hardening_reader(opened)
    monkeypatch.setattr(data, "_module", lambda name: modules[name])
    cfg = RunConfig(run_id="guard", model=tiny_config(), data=_data(root, session))
    try:
        data.open_training_windows(cfg)
    finally:
        monkeypatch.undo()


def test_the_stand_in_reader_trains_on_a_clean_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    session = _trainable(tmp_path / "datasets", "arland_ok")
    opened: list[Path] = []
    _train(monkeypatch, tmp_path / "datasets", "arland_ok", opened)
    assert {p.name for p in opened} == {"meta.json", "state.npy", "slot_id.npy", "latents.npy", "valid.npy"}
    assert all(p.is_relative_to(session.resolve()) for p in opened)


def test_a_redirected_shard_folder_cannot_reach_sealed_latents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Codex round-3 bypass 1: only features_v1 is linked, to another permitted session whose sibling latents link
    into SEALED. The reader derives the latents from the resolved folder, so that is the folder the guard checks."""
    root = tmp_path / "datasets"
    sealed = _trainable(tmp_path / "heldout" / "everon" / "SEALED", "arland_sealed", value=99.0)
    other = _trainable(root, "arland_other")
    (other / "features_v1" / "meta.json").write_text(
        json.dumps(
            {"schema": "drivedit.features_v1-shard/1", "session_id": "arland_ok", "map_id": "arland", "split": "train"}
            | {"decision_hz": 5.0, "arrays": {"state": {"file": "state.npy"}}}
        )
    )
    other_latents = other / "latents" / CODEC
    _empty(other_latents)
    other_latents.symlink_to(sealed / "latents" / CODEC)
    session = _trainable(root, "arland_ok")  # its own latents folder is clean, but nothing reads it
    _empty(session / "features_v1")
    (session / "features_v1").symlink_to(other / "features_v1")
    opened: list[Path] = []
    with pytest.raises(SealedDataError, match="never reads"):
        _train(monkeypatch, root, "arland_ok", opened)
    assert opened == []


@pytest.mark.parametrize("where", ["sealed absolute", "outside absolute", "parent", "subfolder"])
def test_meta_json_cannot_direct_an_array_outside_its_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    """Codex round-3 bypass 2: meta.json names an array file the reader would open outside the shard folder."""
    root = tmp_path / "datasets"
    session = _trainable(root, "arland_ok")
    sealed = _trainable(tmp_path / "heldout" / "everon" / "SEALED", "arland_sealed")
    elsewhere = _trainable(root, "arland_elsewhere")
    file = {
        "sealed absolute": str((sealed / "features_v1" / "state.npy").resolve()),
        "outside absolute": str((elsewhere / "features_v1" / "state.npy").resolve()),
        "parent": "../../arland_elsewhere/features_v1/state.npy",
        "subfolder": "sub/state.npy",
    }[where]
    meta_path = session / "features_v1" / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["arrays"]["state"]["file"] = file
    meta_path.write_text(json.dumps(meta))
    opened: list[Path] = []
    with pytest.raises(TrainingDataError, match="only a plain file name"):
        _train(monkeypatch, root, "arland_ok", opened)
    assert opened == []


@pytest.mark.parametrize("folder", ["features_v1", "latents"])
def test_a_plain_name_must_resolve_inside_its_folder(tmp_path: Path, folder: str) -> None:
    """A plain array or latents file name that is a symlink to a permitted file elsewhere is refused too."""
    root = tmp_path / "datasets"
    session = _trainable(root, "arland_ok")
    elsewhere = _trainable(root, "arland_elsewhere")
    name, target = (
        ("state.npy", elsewhere / "features_v1" / "state.npy")
        if folder == "features_v1"
        else ("valid.npy", elsewhere / "latents" / CODEC / "valid.npy")
    )
    link = session / folder / (CODEC if folder == "latents" else "") / name
    link.unlink()
    link.symlink_to(target)
    with pytest.raises(TrainingDataError, match="outside"):
        check_training_sessions(_data(root, "arland_ok"), 5.0)


@pytest.mark.parametrize(("key", "name"), [("sha256", "../latents.npy"), ("aligned_to", "/data/slot_id.npy")])
def test_the_latents_meta_names_only_plain_files(tmp_path: Path, key: str, name: str) -> None:
    root = tmp_path / "datasets"
    session = _trainable(root, "arland_ok")
    meta = _latents_meta(session)
    if key == "sha256":
        meta["sha256"][name] = "x"
    else:
        meta["aligned_to"]["file"] = name
    _write_latents_meta(session, meta)
    with pytest.raises(TrainingDataError, match="only a plain file name"):
        check_training_sessions(_data(root, "arland_ok"), 5.0)


def _replace_with_hard_link(path: Path, target: Path) -> None:
    path.unlink()
    os.link(target, path)


@pytest.mark.parametrize("resource", ["features_v1/meta.json", "features_v1/state.npy", f"latents/{CODEC}/latents.npy"])
def test_a_hard_link_to_sealed_data_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource: str) -> None:
    """Codex round-4 HIGH: a permitted path hard-linked to a file under SEALED shares its inode (st_nlink 2), so no
    path rule can see it. A training file must have exactly one link."""
    root = tmp_path / "datasets"
    session = _trainable(root, "arland_ok")
    sealed = _trainable(tmp_path / "heldout" / "everon" / "SEALED", "arland_sealed", value=99.0)
    target = sealed / resource
    if resource.endswith("meta.json"):  # a sealed meta that would pass every metadata check if it were read
        meta = json.loads((session / resource).read_text())
        target.write_text(json.dumps(meta))
    _replace_with_hard_link(session / resource, target)
    assert (session / resource).stat().st_nlink == 2
    opened: list[Path] = []
    with pytest.raises(TrainingDataError, match="2 hard links"):
        _train(monkeypatch, root, "arland_ok", opened)
    assert opened == []


def test_an_array_symlinked_within_its_folder_is_refused(tmp_path: Path) -> None:
    """It resolves inside the folder, but only a regular file passes."""
    session = _trainable(tmp_path, "arland_ok")
    (session / "features_v1" / "state.npy").unlink()
    (session / "features_v1" / "state.npy").symlink_to(session / "features_v1" / "slot_id.npy")
    with pytest.raises(TrainingDataError, match="not a regular file"):
        check_training_sessions(_data(tmp_path, "arland_ok"), 5.0)


@pytest.mark.parametrize("name", ["latents\x00.npy", "state\x01.npy", "valid\x7f.npy"])
def test_control_characters_are_not_plain_names(tmp_path: Path, name: str) -> None:
    session = _trainable(tmp_path, "arland_ok")
    meta = _latents_meta(session)
    meta["sha256"] = {name: "x"}
    _write_latents_meta(session, meta)
    with pytest.raises(TrainingDataError, match="only a plain file name"):
        check_training_sessions(_data(tmp_path, "arland_ok"), 5.0)


def test_latents_meta_names_get_the_full_resource_check(tmp_path: Path) -> None:
    """Codex round-4 LOW: a plain name in the latents meta, symlinked outside the latents folder, or missing."""
    root = tmp_path / "datasets"
    session = _trainable(root, "arland_ok")
    elsewhere = _trainable(root, "arland_elsewhere")
    latents = session / "latents" / CODEC
    (latents / "extra.npy").symlink_to(elsewhere / "latents" / CODEC / "valid.npy")
    meta = _latents_meta(session)
    cases = [
        ({"extra.npy": "y"}, "slot_id.npy", "outside"),
        ({"gone.npy": "y"}, "slot_id.npy", "missing"),
        ({}, "gone.npy", "missing"),  # aligned_to.file names a file of the shard folder
    ]
    for named, aligned_file, match in cases:
        meta["sha256"], meta["aligned_to"]["file"] = named, aligned_file
        _write_latents_meta(session, meta)
        with pytest.raises(TrainingDataError, match=match):
            check_training_sessions(_data(root, "arland_ok"), 5.0)


# ------------------------------------------------------------------------------------- whose latents these are


@pytest.mark.parametrize(
    ("donor", "error", "match"),
    [("arland_other", TrainingDataError, "not of this shard"), ("everon_source", SealedDataError, "sealed test map")],
)
def test_latents_copied_from_another_session_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, donor: str, error: type[Exception], match: str
) -> None:
    """Codex round-5 HIGH: a complete encoder output, meta.json and all, copied into another session's latents
    folder. Every file is a fresh regular file with one link, and the slot_id.npy hashes match, so only the latents
    meta's session_id tells; an Everon donor is sealed data."""
    root = tmp_path / "datasets"
    session = _trainable(root, "arland_ok")
    source = _trainable(tmp_path / "encoded", donor, value=99.0)
    assert _latents_meta(source)["aligned_to"]["sha256"] == _latents_meta(session)["aligned_to"]["sha256"]
    shutil.rmtree(session / "latents" / CODEC)
    shutil.copytree(source / "latents" / CODEC, session / "latents" / CODEC)
    opened: list[Path] = []
    with pytest.raises(error, match=match):
        _train(monkeypatch, root, "arland_ok", opened)
    assert opened == []


@pytest.mark.parametrize("aligned", [{"file": "slot_id.npy", "sha256": "0" * 64}, {"file": "slot_id.npy"}])
def test_latents_of_another_build_of_the_session_are_refused(tmp_path: Path, aligned: dict[str, str]) -> None:
    """The session id matches, but the latents were encoded from another slot_id.npy (or do not say which)."""
    session = _trainable(tmp_path, "arland_ok")
    meta = _latents_meta(session)
    meta["aligned_to"] = aligned
    _write_latents_meta(session, meta)
    with pytest.raises(TrainingDataError, match=r"aligned_to\.sha256"):
        check_training_sessions(_data(tmp_path, "arland_ok"), 5.0)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("session_id", None, TrainingDataError),
        ("codec_id", "flux2-vae", TrainingDataError),
        ("map_id", "kolguyev", TrainingDataError),
        ("split", "val_l", TrainingDataError),
        ("map_id", "Everon", SealedDataError),
        ("split", "everon_test", SealedDataError),
    ],
)
def test_the_latents_meta_must_agree_with_the_shard(
    tmp_path: Path, key: str, value: str | None, error: type[Exception]
) -> None:
    session = _trainable(tmp_path, "arland_ok")
    meta = _latents_meta(session)
    meta[key] = value
    _write_latents_meta(session, meta)
    with pytest.raises(error):
        check_training_sessions(_data(tmp_path, "arland_ok"), 5.0)
    meta[key] = {"session_id": "arland_ok", "codec_id": CODEC, "map_id": "arland", "split": "train"}[key]
    _write_latents_meta(session, meta)
    assert check_training_sessions(_data(tmp_path, "arland_ok"), 5.0)  # the agreeing value passes
