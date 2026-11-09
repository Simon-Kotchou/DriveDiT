"""drivedit.codec.dinov3: the DINOv3 checkout is verified file by file, never fetched, and nothing else is imported.

The refusals run anywhere. The load of the real checkout runs in a subprocess (it registers ``dinov3`` modules) and
skips when no verified checkout is present (CI): ``DRIVEDIT_DINOV3_DIR`` or the default directory under CODE_ROOT of the
local config (DD-22).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from drivedit.codec import dinov3
from drivedit.codec.dinov3 import (
    DINOV3_COMMIT,
    DINOV3_DIR_ENV,
    DINOV3_SHA256,
    DinoV3SourceError,
    dinov3_dir,
    load_dinov3_vitb16,
    verify_checkout,
)

SRC = Path(__file__).resolve().parents[3] / "src"
LOCAL_CONFIG_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "local_config" / "placeholder.local.env"


def _verified_checkout() -> Path | None:
    try:
        return verify_checkout(dinov3_dir())
    except DinoV3SourceError:
        return None


CHECKOUT = _verified_checkout()


@pytest.fixture(autouse=True)
def no_loaded_dinov3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide the dinov3 modules an earlier test loaded (test_pretrained builds the real CADRE); restored afterwards."""
    for name in [name for name in sys.modules if name.split(".")[0] == "dinov3"]:
        monkeypatch.delitem(sys.modules, name)


def test_the_directory_is_keyword_then_environment_then_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DINOV3_DIR_ENV, raising=False)
    monkeypatch.setenv("DRIVEDIT_LOCAL_CONFIG", str(LOCAL_CONFIG_FIXTURE))
    assert dinov3_dir() == Path("/mnt/x/CodeRoot/third_party/dinov3")
    monkeypatch.setenv("DRIVEDIT_LOCAL_CONFIG", str(LOCAL_CONFIG_FIXTURE.parent / "missing.env"))
    with pytest.raises(DinoV3SourceError, match=f"set {DINOV3_DIR_ENV}"):
        dinov3_dir()
    monkeypatch.setenv(DINOV3_DIR_ENV, "/elsewhere/dinov3")
    assert dinov3_dir() == Path("/elsewhere/dinov3")
    assert dinov3_dir("/given") == Path("/given")


def test_a_missing_checkout_names_the_fetch_command_and_fetches_nothing(tmp_path: Path) -> None:
    with pytest.raises(DinoV3SourceError, match=f"Fetch it with: git init .* {DINOV3_COMMIT}"):
        load_dinov3_vitb16(tmp_path / "absent")
    assert not (tmp_path / "absent").exists()


def test_a_modified_file_is_refused(tmp_path: Path) -> None:
    for relative in DINOV3_SHA256:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# not DINOv3\n", encoding="utf-8")
    with pytest.raises(DinoV3SourceError, match="sha256 mismatch"):
        load_dinov3_vitb16(tmp_path)


def test_a_foreign_dinov3_module_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    foreign = types.ModuleType("dinov3")
    monkeypatch.setitem(sys.modules, "dinov3", foreign)
    with pytest.raises(DinoV3SourceError, match="not loaded from the verified checkout"):
        load_dinov3_vitb16(tmp_path)


def test_the_manifest_is_the_backbone_path() -> None:
    assert "dinov3/hub/backbones.py" in DINOV3_SHA256
    assert "dinov3/models/vision_transformer.py" in DINOV3_SHA256
    assert not any("detectors" in name or "segmentors" in name for name in DINOV3_SHA256)
    assert dinov3.PACKAGE == "dinov3"


@pytest.mark.skipif(
    CHECKOUT is None, reason=f"no verified DINOv3 checkout ({DINOV3_DIR_ENV} or <CODE_ROOT>/third_party)"
)
def test_the_verified_checkout_builds_vitb16_from_manifest_files_only(tmp_path: Path) -> None:
    assert CHECKOUT is not None
    copy = tmp_path / "dinov3"
    for relative in DINOV3_SHA256:
        (copy / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CHECKOUT / relative, copy / relative)
    code = (
        "import sys, torch\n"
        "from drivedit.codec.dinov3 import DINOV3_SHA256, load_dinov3_vitb16\n"
        f"model = load_dinov3_vitb16({str(copy)!r})\n"
        "assert type(model).__name__ == 'DinoVisionTransformer'\n"
        "assert sum(p.numel() for p in model.parameters()) == 85669632\n"
        "assert not any(p.requires_grad for p in model.parameters())\n"
        "mods = [m for n, m in sys.modules.items() if n.split('.')[0] == 'dinov3']\n"
        "files = {m.__file__ for m in mods if getattr(m, '__file__', None)}\n"
        f"assert files == {{str({str(copy)!r} + '/' + r) for r in DINOV3_SHA256}}, files\n"
        "assert all(m.__path__ == [] for n, m in sys.modules.items() if n.split('.')[0] == 'dinov3'"
        " and hasattr(m, '__path__'))\n"
        "try:\n"
        "    import dinov3.hub.detectors\n"
        "except ImportError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('a file outside the manifest was imported')\n"
        "x = torch.zeros(1, 3, 288, 512)\n"
        "assert model.get_intermediate_layers(x, n=(2, 11), norm=True, reshape=True)[0].shape == (1, 768, 18, 32)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}  # inherit TMPDIR, CUDA_VISIBLE_DEVICES, HF_HUB_OFFLINE, ...
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
