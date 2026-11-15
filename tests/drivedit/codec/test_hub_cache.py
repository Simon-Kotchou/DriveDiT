"""``fetch_weights`` downloads only into a Hugging Face cache inside the write boundary, and huggingface_hub is imported
late enough for an entry point to route that cache first (DD-23). Nothing downloads: ``snapshot_download`` is replaced,
and the boundary is a DD-22 fake world."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub import constants

from drivedit.codec.base import CODECS, HUB_WRITE_DIRS, fetch_weights
from drivedit.local_paths import WriteRefusedError
from tests.drivedit.boundary import NEVER, assert_never_written, world

SOURCE = CODECS["wan21-vae"].weights
SRC = Path(__file__).resolve().parents[3] / "src"


@pytest.fixture
def downloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[dict[str, object]]:
    """``huggingface_hub.snapshot_download`` replaced by a stub that records its call and returns a complete snapshot
    in ``tmp_path``."""
    snapshot = tmp_path / "snapshot"
    for name in SOURCE.files:
        (snapshot / name).parent.mkdir(parents=True, exist_ok=True)
        (snapshot / name).write_bytes(b"")
    calls: list[dict[str, object]] = []

    def stub(repo_id: str, **kwargs: object) -> str:
        calls.append({"repo_id": repo_id, **kwargs})
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", stub)
    return calls


def _point_cache(monkeypatch: pytest.MonkeyPatch, hub: str, xet: str) -> None:
    monkeypatch.setattr(constants, "HF_HUB_CACHE", hub)
    monkeypatch.setattr(constants, "HF_XET_CACHE", xet)


def test_the_checked_dirs_are_the_ones_a_download_writes() -> None:
    assert HUB_WRITE_DIRS == ("HF_HUB_CACHE", "HF_XET_CACHE")
    assert all(isinstance(getattr(constants, name), str) for name in HUB_WRITE_DIRS)


@pytest.mark.parametrize(
    ("hub", "xet", "refused"),
    [
        (
            NEVER + "/hub",
            "{B}/shm/xet",
            "HF_HUB_CACHE=" + re.escape(NEVER),
        ),  # the library default lies on the root filesystem too
        ("{B}/shm/hub", "{M}/z/xet", "HF_XET_CACHE=.*Windows system drive"),
        ("{M}/y/hub", "{M}/x/DataRoot/xet", "HF_HUB_CACHE=.*not on the data drive"),
    ],
)
def test_a_cache_outside_the_boundary_is_refused_before_any_download(
    monkeypatch: pytest.MonkeyPatch, downloads: list[dict[str, object]], hub: str, xet: str, refused: str
) -> None:
    with world() as w:
        _point_cache(monkeypatch, w.path(hub), w.path(xet))
        with pytest.raises(WriteRefusedError, match=refused):
            fetch_weights(SOURCE)
    assert downloads == []
    assert_never_written()


def test_a_cache_inside_the_boundary_downloads_as_before(
    monkeypatch: pytest.MonkeyPatch, downloads: list[dict[str, object]]
) -> None:
    with world() as w:
        _point_cache(monkeypatch, f"{w.mnt}/x/DataRoot/cache/hub", f"{w.shm}/xet")
        snapshot = fetch_weights(SOURCE)
    assert [call["repo_id"] for call in downloads] == [SOURCE.repo_id]
    assert downloads[0]["revision"] == SOURCE.revision
    assert "cache_dir" not in downloads[0]  # huggingface_hub's own resolution, unchanged
    assert snapshot.name == "snapshot"


@pytest.mark.parametrize(
    ("link", "target"),
    [
        (".locks", "{M}/z"),  # the lock dir huggingface_hub creates a repo's lock folder in
        ("models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/blobs", "{M}/y/blobs"),  # where the weights land
        ("models--x--y/blobs/abc.incomplete", NEVER + "/abc"),  # a stale partial download, dangling
    ],
)
def test_a_link_in_the_cache_that_leads_out_is_refused(
    monkeypatch: pytest.MonkeyPatch, downloads: list[dict[str, object]], link: str, target: str
) -> None:
    with world() as w:
        hub = Path(w.mnt, "x", "DataRoot", "cache", "hub")
        (hub / link).parent.mkdir(parents=True, exist_ok=True)
        (hub / link).symlink_to(w.path(target))
        _point_cache(monkeypatch, str(hub), f"{w.shm}/xet")
        with pytest.raises(WriteRefusedError, match=f"HF_HUB_CACHE=.*the link .*{link} leads to"):
            fetch_weights(SOURCE)
    assert downloads == []
    assert_never_written()


def test_a_link_out_through_a_linked_directory_is_refused(
    monkeypatch: pytest.MonkeyPatch, downloads: list[dict[str, object]]
) -> None:
    """``.locks`` leads to an allowed RAM dir, and a link there leads out: huggingface_hub would follow both."""
    with world() as w:
        hub = Path(w.mnt, "x", "DataRoot", "cache", "hub")
        hub.mkdir(parents=True)
        external = Path(w.shm, "external-locks")
        external.mkdir()
        (hub / ".locks").symlink_to(external)
        (external / "models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers").symlink_to(NEVER)
        _point_cache(monkeypatch, str(hub), f"{w.shm}/xet")
        with pytest.raises(
            WriteRefusedError, match=f"models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers leads to {re.escape(NEVER)}"
        ):
            fetch_weights(SOURCE)
    assert downloads == []
    assert_never_written()


def test_links_inside_the_cache_are_fine(monkeypatch: pytest.MonkeyPatch, downloads: list[dict[str, object]]) -> None:
    with world() as w:
        hub = Path(w.mnt, "x", "DataRoot", "cache", "hub")
        snapshot = hub / "models--a--b" / "snapshots" / "0"
        (hub / "models--a--b" / "blobs").mkdir(parents=True)
        (hub / "models--a--b" / "blobs" / "f").write_bytes(b"")
        snapshot.mkdir(parents=True)
        (snapshot / "f").symlink_to("../../blobs/f")  # the layout huggingface_hub writes
        _point_cache(monkeypatch, str(hub), f"{w.shm}/xet")
        fetch_weights(SOURCE)
    assert len(downloads) == 1


def test_an_xet_log_outside_the_boundary_is_refused(
    monkeypatch: pytest.MonkeyPatch, downloads: list[dict[str, object]]
) -> None:
    with world() as w:
        _point_cache(monkeypatch, f"{w.shm}/hub", f"{w.shm}/xet")
        monkeypatch.setenv("HF_XET_LOG_DEST", NEVER + "/xet.log")
        with pytest.raises(WriteRefusedError, match=f"HF_XET_LOG_DEST={re.escape(NEVER)}/"):
            fetch_weights(SOURCE)
    assert downloads == []
    assert_never_written()


def test_local_only_reads_and_is_not_checked(
    monkeypatch: pytest.MonkeyPatch, downloads: list[dict[str, object]]
) -> None:
    _point_cache(monkeypatch, NEVER + "/hub", NEVER + "/xet")
    fetch_weights(SOURCE, local_only=True)
    assert downloads[0]["local_files_only"] is True
    assert_never_written()


def test_importing_the_codec_package_leaves_huggingface_hub_unimported(tmp_path: Path) -> None:
    """So an entry point's prepare_process runs before huggingface_hub fixes its cache paths."""
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import drivedit.codec, drivedit.codec.encode; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] in ('huggingface_hub', 'diffusers')))"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(SRC)],
        cwd=tmp_path,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
