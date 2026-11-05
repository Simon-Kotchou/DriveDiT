"""The CADRE arm's source pinning (CI) and, with the pinned checkout, the import of CADRE's own module (local).

The full CADRE codec needs the CADRE checkout, DINOv3's torch.hub code and 2.2 GB of weights; it is exercised in
test_pretrained.py, which skips without them.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
import torch

from drivedit.codec import cadre
from drivedit.codec.cadre import CADRE_SOURCE_ENV, CADRE_SOURCE_SHA256, CadreSourceError, import_cadre

CHECKOUT = os.environ.get(CADRE_SOURCE_ENV)
needs_checkout = pytest.mark.skipif(not CHECKOUT, reason=f"{CADRE_SOURCE_ENV} (a pinned CADRE checkout) is not set")


def test_load_without_a_checkout_raises_before_downloading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CADRE_SOURCE_ENV, raising=False)

    def no_download(*args: object, **kwargs: object) -> None:
        raise AssertionError("downloaded before checking the CADRE checkout")

    monkeypatch.setattr(cadre, "fetch_weights", no_download)
    with pytest.raises(CadreSourceError, match=CADRE_SOURCE_ENV):
        cadre.load("cadre-1x16x16", device=torch.device("cpu"), dtype=torch.float32)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CadreSourceError, match="missing"):
        import_cadre(tmp_path)


def test_a_modified_checkout_is_refused(tmp_path: Path) -> None:
    for relative in CADRE_SOURCE_SHA256:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# not CADRE\n", encoding="utf-8")
    with pytest.raises(CadreSourceError, match="sha256 mismatch"):
        import_cadre(tmp_path)


@needs_checkout
def test_the_pinned_checkout_imports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert CHECKOUT is not None
    copy = tmp_path / "cadre"
    for relative in CADRE_SOURCE_SHA256:
        (copy / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(CHECKOUT) / relative, copy / relative)
    for name in ("util", "util.hub", "util.attention", "model_autoencoder"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = import_cadre(copy)
    assert {"cadre_1x16x16", "cadre_1x48x48"} <= set(module.CADRE_MODELS)
    assert module.CADRE_MODELS["cadre_1x16x16"].latent_grid == (18, 32)
    assert module.CADRE_MODELS["cadre_1x48x48"].latent_grid == (6, 11)
    assert import_cadre(copy) is module  # a second call reuses the loaded module
    with pytest.raises(CadreSourceError, match="already imported"):
        import_cadre(Path(CHECKOUT))  # same files, another location
