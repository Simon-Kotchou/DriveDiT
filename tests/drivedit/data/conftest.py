"""Shared fixtures of the drivedit.data tests: the features_v1 spec, the golden synthetic dc-2.0 session
(``tests/fixtures/capture_dc2``) read once, its shard built once, and scratch directories.

Scratch space lies in validated RAM: $TMPDIR when that is on /dev/shm/, else /dev/shm/ (never /tmp;
tests/fixtures/local_config/scratch.py). Every test reads the synthetic local config (DD-22: data root
/mnt/x/DataRoot, system drive /mnt/z, neither a real mount), so no default root names a real drive. No torch here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from drivedit.data import build
from drivedit.data.shard import Shard
from drivedit.data.spec import FeatureSpec, load_spec

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "tests" / "fixtures" / "capture_dc2"
GOLDEN_ID = "arland_T0b_20261025T091512Z_wb_i00_5e00"
GOLDEN_DIR = FIXTURES / "golden" / GOLDEN_ID
GOLDEN_JSON = FIXTURES / "golden.json"
ROUTES_DIR = FIXTURES / "routes"
CARD = FIXTURES / "cards" / "S105_synthetic.json"
LOCAL_CONFIG_FIXTURE = REPO / "tests" / "fixtures" / "local_config" / "placeholder.local.env"
_scratch_spec = importlib.util.spec_from_file_location(
    "dd22_scratch", REPO / "tests" / "fixtures" / "local_config" / "scratch.py"
)
assert _scratch_spec is not None
assert _scratch_spec.loader is not None
_scratch = importlib.util.module_from_spec(_scratch_spec)
_scratch_spec.loader.exec_module(_scratch)


@pytest.fixture(autouse=True)
def placeholder_local_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """The synthetic local config for every test here (DD-22): the defaults of drivedit.data.build come from it."""
    monkeypatch.setenv("DRIVEDIT_LOCAL_CONFIG", str(LOCAL_CONFIG_FIXTURE))


def scratch_base() -> Path:
    """The validated RAM scratch base (tests/fixtures/local_config/scratch.py): the builder writes through the real
    write boundary, so these tests skip where it refuses RAM."""
    return Path(_scratch.boundary_base())


@pytest.fixture(scope="session")
def scratch() -> Iterator[Path]:
    """One scratch directory for the whole test session, removed at the end."""
    path = Path(tempfile.mkdtemp(prefix="dd11_data_", dir=scratch_base()))
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def tmp(scratch: Path, request: pytest.FixtureRequest) -> Path:
    """A fresh directory per test under the session scratch."""
    prefix = re.sub(r"[^A-Za-z0-9_]", "_", request.node.name)[:40]  # parametrized ids may hold '/' or ':'
    return Path(tempfile.mkdtemp(prefix=prefix + "_", dir=scratch))


@pytest.fixture(scope="session")
def spec() -> FeatureSpec:
    return load_spec()


@pytest.fixture(scope="session")
def golden_session() -> Any:
    return build.open_dc2_session(GOLDEN_DIR, vehicle_card=CARD)


@pytest.fixture(scope="session")
def golden_shard_dir(golden_session: Any, spec: FeatureSpec, scratch: Path) -> Path:
    root = scratch / "golden_root"
    return build.write_shard(golden_session, root, spec=spec, routes_dir=ROUTES_DIR, dataset_roots=(root,))


@pytest.fixture(scope="session")
def golden_truth() -> dict[str, Any]:
    """golden.json: the decision samples make_synthetic.py computed from its own simulation (not by the reader)."""
    truth: dict[str, Any] = json.loads(GOLDEN_JSON.read_text("utf-8"))
    return truth


@pytest.fixture(scope="session")
def synth() -> ModuleType:
    """tests/fixtures/capture_dc2/make_synthetic.py, loaded by file path like the reader tests load it."""
    path = FIXTURES / "make_synthetic.py"
    module_spec = importlib.util.spec_from_file_location("dc2_make_synthetic", path)
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


@pytest.fixture
def shard_copy(golden_shard_dir: Path, tmp: Path) -> Path:
    """A writable copy of the golden shard at <tmp>/<session_id>/features_v1."""
    dst = tmp / golden_shard_dir.parent.name / "features_v1"
    shutil.copytree(golden_shard_dir, dst)
    return dst


def write_latents(shard: Shard, codec_id: str, *, channels: int = 2, hw: tuple[int, int] = (3, 4)) -> np.ndarray:
    """A synthetic latents directory (DRIVEDIT_INTERFACES section 3): latents[t, c] = t + 0.25 c where slot_id >= 0,
    zeros elsewhere. Returns the float16 array."""
    root = shard.path.parent / "latents" / codec_id
    root.mkdir(parents=True)
    slot = np.asarray(shard.array("slot_id"))
    valid = slot >= 0
    t = np.arange(shard.n_steps, dtype=np.float32)[:, None, None, None]
    c = 0.25 * np.arange(channels, dtype=np.float32)[None, :, None, None]
    latents = np.broadcast_to(t + c, (shard.n_steps, channels, *hw)).astype(np.float16)
    latents[~valid] = 0
    np.save(root / "latents.npy", latents)
    np.save(root / "valid.npy", valid)
    meta = {
        "schema": "drivedit.latents/1",
        "codec_id": codec_id,
        "session_id": shard.session_id,
        "aligned_to": {"file": "slot_id.npy", "sha256": shard.meta["arrays"]["slot_id"]["sha256"]},
        "weights": {"source": "synthetic test", "revision": "none"},
        "input_size": [288, 512],
        "latent_shape": [channels, *hw],
        "mean": [0.0] * channels,
        "std": [1.0] * channels,
        "dtype": "float16",
        "encoded_utc": "2026-11-05T00:00:00Z",
    }
    (root / "meta.json").write_text(json.dumps(meta), "utf-8")
    return latents


def rewrite_array(shard_dir: Path, name: str, array: np.ndarray) -> None:
    """Replace one array of a shard copy and keep its meta.json sha256 true."""
    path = shard_dir / f"{name}.npy"
    np.save(path, array)
    meta = json.loads((shard_dir / "meta.json").read_text("utf-8"))
    meta["arrays"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (shard_dir / "meta.json").write_text(json.dumps(meta), "utf-8")
