"""tools/codec/measure_codecs.py writes only where the write boundary allows (DD-22 round 2): --out is checked first,
before CUDA, weights or frames are touched, and every file is narrowed to it."""

from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]
FIXTURE = REPO / "tests" / "fixtures" / "local_config" / "placeholder.local.env"


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def measure(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("DRIVEDIT_LOCAL_CONFIG", str(FIXTURE))
    return _load("dd19_measure_codecs", REPO / "tools" / "codec" / "measure_codecs.py")


@pytest.mark.parametrize("out", ["/tmp/dd22_codec_never_written", "/var/dd22_codec_never_written"])
def test_out_off_the_boundary_is_refused_first(measure: ModuleType, out: str) -> None:
    with pytest.raises(SystemExit, match="refusing to write"):
        measure.main(["--deriv", "/dev/shm/dd22_no_such_deriv", "--out", out])
    assert not os.path.exists(out)


def test_files_are_narrowed_to_the_checked_out(measure: ModuleType) -> None:
    out = Path("/dev/shm") / f"dd22_codec_out_{os.getpid()}"
    measure._OUT_ROOT[:] = [out]
    try:
        assert measure.checked(out / "recon.png") == out / "recon.png"
        with pytest.raises(SystemExit, match="outside"):
            measure.checked(Path("/dev/shm") / "dd22_codec_elsewhere.png")
        with pytest.raises(SystemExit, match=re.escape("'..'")):
            measure.checked(out / ".." / "escape.png")
    finally:
        measure._OUT_ROOT.clear()
    assert not out.exists()
