"""The write boundary (DD-22 round 2): one table of vectors (tests/fixtures/local_config/boundary_vectors.py) through
both entry points, tools/local_config.py check_write and drivedit.local_paths.check_write, in a fake world under
/dev/shm (fake mountinfo, fake drives). The boundary only reads; nothing here writes outside the world.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FIXTURES = os.path.join(HERE, "fixtures", "local_config")


def _load(name, path):
    mod = sys.modules.get(name)
    if mod is not None and os.path.realpath(getattr(mod, "__file__", "") or "") == os.path.realpath(path):
        return mod
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


LC_PATH = os.path.realpath(os.path.join(REPO, "tools", "local_config.py"))
lc = _load("dd_local_config", LC_PATH)
bv = _load("dd22_boundary_vectors", os.path.join(FIXTURES, "boundary_vectors.py"))
evidence = _load("dd_test_evidence", os.path.join(FIXTURES, "evidence.py"))


def _package_check(path, roots):
    from drivedit import local_paths

    assert local_paths._loader() is lc  # one implementation behind both entry points
    try:
        return local_paths.check_write(path, roots)
    except local_paths.WriteRefusedError as exc:
        raise lc.WriteRefused(str(exc)) from exc


ENTRY_POINTS = {"tools": lc.check_write, "package": _package_check}


@pytest.fixture(scope="module")
def world():
    with bv.world() as w:
        yield w


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
@pytest.mark.parametrize("state", sorted(bv.STATES))
def test_vectors(world, state, entry):
    check = ENTRY_POINTS[entry]
    wrong = []
    with world.use(lc, state):
        for vid, dest, roots, allowed_in, fragment in bv.VECTORS:
            path = world.path(dest)
            rts = None if roots is None else [world.path(r) for r in roots]
            try:
                got = check(path, rts)
                outcome, msg = True, got
            except lc.WriteRefused as exc:
                outcome, msg = False, str(exc)
            want = state in allowed_in
            if outcome != want:
                wrong.append("%s: %s, want %s (%s)" % (vid, "allowed" if outcome else "refused",
                                                       "allowed" if want else "refused", msg))
            elif not outcome and fragment and fragment not in msg:
                wrong.append("%s: refusal %r lacks %r" % (vid, msg, fragment))
            elif outcome and not os.path.realpath(path).startswith(world.base):
                wrong.append("%s: allowed outside the world: %s" % (vid, msg))
    assert not wrong, "\n".join(wrong)
    assert not os.path.exists("/tmp/dd22-boundary-vector-never-written")


def test_vectors_cover_every_required_case():
    ids = {v[0] for v in bv.VECTORS}
    for need in ("system_drive", "unmounted_drive", "tmp", "rootfs", "widen_to_tmp", "widen_to_root",
                 "symlink_leaf", "dotdot_escape", "nested_mount", "device_mismatch", "symlink_dir_to_system"):
        assert need in ids
    assert {"real_like", "placeholder", "missing", "malformed", "marked_data", "swapped",
            "wrong_system_drive", "unmounted_data"} <= set(bv.STATES)


def test_allowed_destination_is_resolved(world):
    with world.use(lc, "real_like"):
        assert lc.check_write(world.shm + "/dir_in_ram/x.json") == world.shm + "/real/x.json"
        assert lc.check_write(world.mnt + "/x/DataRoot/a/b.json") == world.mnt + "/x/DataRoot/a/b.json"


def test_system_drive_is_advisory_only(world):
    # SYSTEM_DRIVE names the data drive: writes there still pass (the boundary verified that volume itself), and the
    # real system volume is refused although the config does not name it.
    with world.use(lc, "wrong_system_drive"):
        assert lc.write_refusal(world.mnt + "/x/DataRoot/out.json") is None
        why = lc.write_refusal(world.mnt + "/z/out.json")
        assert "Windows system drive" in why and "names /mnt/x as the system drive" in why


def test_data_root_on_system_volume_refuses_every_drive_write(world):
    with world.use(lc, "swapped"):
        assert "DATA_ROOT is on the Windows system drive" in lc.write_refusal(world.mnt + "/x/DataRoot/out.json")
        assert lc.write_refusal(world.shm + "/out.json") is None


def test_unlistable_root_counts_as_system_volume(world, monkeypatch):
    with world.use(lc, "real_like"):
        real_scandir = os.scandir

        def failing(path):
            if path == world.mnt + "/x":
                raise PermissionError(13, "Permission denied")
            return real_scandir(path)

        monkeypatch.setattr(lc.os, "scandir", failing)
        why = lc.write_refusal(world.mnt + "/x/DataRoot/out.json")
        assert "cannot be listed" in why


@pytest.mark.parametrize("name,ok", bv.CHILD_NAMES)
def test_child_names(world, name, ok):
    with world.use(lc, "real_like"):
        try:
            got = lc.check_child(world.shm, name)
        except lc.WriteRefused as exc:
            assert not ok, str(exc)
            return
        assert ok and got == os.path.join(world.shm, name)


def test_open_new_never_follows_or_overwrites(world):
    with world.use(lc, "real_like"):
        path = lc.write_new(world.shm + "/new_once.txt", b"one\n")
        with pytest.raises(FileExistsError):
            lc.write_new(path, b"two\n")
        assert open(path, "rb").read() == b"one\n"
        with pytest.raises(lc.WriteRefused, match="symlink"):
            lc.open_new(world.shm + "/leaf_link")
        assert open(world.shm + "/real/target").read() == "target\n"


def test_no_config_allows_ram_only(world):
    with world.use(lc, "missing"):
        why = lc.write_refusal(world.mnt + "/x/DataRoot/out.json")
        assert "no data drive is known" in why
        assert lc.write_refusal(world.shm + "/out.json") is None


def test_cli_check_write():
    env = dict(os.environ)
    ok = subprocess.run([sys.executable, "-I", "-B", LC_PATH, "check-write", "/dev/shm/dd22-cli-probe/x.json"],
                        capture_output=True, text=True, env=env, timeout=60)
    assert (ok.returncode, ok.stdout.strip()) == (0, "/dev/shm/dd22-cli-probe/x.json")
    bad = subprocess.run([sys.executable, "-I", "-B", LC_PATH, "check-write", "/tmp/dd22-cli-probe/x.json"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert bad.returncode == 1 and "refusing to write /tmp/dd22-cli-probe/x.json" in bad.stderr
    narrow = subprocess.run([sys.executable, "-I", "-B", LC_PATH, "check-write", "/dev/shm/a/x.json", "--root",
                             "/dev/shm/b"], capture_output=True, text=True, env=env, timeout=60)
    assert narrow.returncode == 1 and "outside /dev/shm/b" in narrow.stderr
    assert not os.path.exists("/dev/shm/dd22-cli-probe")


def test_real_mountinfo_ram_is_verified():
    # The real /dev/shm is a tmpfs: RAM stays writable with the synthetic config (whose drives are mounted nowhere).
    assert lc.write_refusal("/dev/shm/dd22-boundary-probe/x") is None
    assert "refusing" in lc.write_refusal("/tmp/dd22-boundary-probe/x")


def test_opt_in_scratch_is_validated_before_anything_is_created(world, monkeypatch):
    """Codex row 1: DC_TEST_E_SCRATCH is checked (resolved dir, the boundary with the machine's config, its data
    drive, apart from DATA_ROOT) before a test creates anything in it."""
    ev_lc = evidence._loader()
    monkeypatch.setenv("DD_TEST_LOCAL_CONFIG", bv.PLACEHOLDER)
    for scratch in ("/tmp", "/dev/shm", "/var"):   # the Linux disk, RAM, a data drive mounted nowhere
        monkeypatch.setenv(evidence.SCRATCH_ENV, scratch)
        got, why = evidence.data_drive_scratch()
        assert got is None and why, scratch
    with world.use(ev_lc, "real_like"):
        monkeypatch.setenv("DD_TEST_LOCAL_CONFIG", world.config_path("real_like"))
        cases = {world.mnt + "/x/DataRoot": "overlaps", world.mnt + "/x/DataRoot/nested": "refusing",
                 world.mnt + "/x": "overlaps", world.mnt + "/z": "Windows system drive",
                 world.mnt + "/y/DataRoot": "refusing", world.shm: "not on the data drive",
                 world.shm + "/dir_in_ram": "symlink", world.mnt + "/x/elsewhere": None}
        for scratch, fragment in cases.items():
            monkeypatch.setenv(evidence.SCRATCH_ENV, scratch)
            got, why = evidence.data_drive_scratch()
            if fragment is None:
                assert (got, why) == (scratch, None)
            else:
                assert got is None and fragment in why, (scratch, why)
    monkeypatch.delenv("DD_TEST_LOCAL_CONFIG")
    assert evidence.data_drive_scratch()[0] is None
