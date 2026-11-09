"""Opt-in machine evidence for the tests (DD-22).

A few tests compare repo files with documents or data of a real machine (the capture program, the real legacy and
captured sessions, the official script dump) or run on a real drvfs scratch directory. They run only when
DD_TEST_LOCAL_CONFIG names that machine's drivedit.local.env; everywhere else, CI included, they skip. Every other
test reads the synthetic config placeholder.local.env next to this file and names no real path.

    spec = importlib.util.spec_from_file_location("dd_test_evidence", "<repo>/tests/fixtures/local_config/evidence.py")
"""

import importlib.util
import os

ENV = "DD_TEST_LOCAL_CONFIG"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
FIXTURE = os.path.join(HERE, "placeholder.local.env")
LOADER = os.path.join(REPO, "tools", "local_config.py")


_LC = []


def _loader():
    """tools/local_config.py as a module of its own (loaded once), so the machine's config never leaks into the
    module the code under test uses."""
    if not _LC:
        spec = importlib.util.spec_from_file_location("dd_test_evidence_local_config", LOADER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LC.append(mod)
    return _LC[0]


def machine_config_path():
    """The file DD_TEST_LOCAL_CONFIG names, or None (the opt-in is off)."""
    path = os.environ.get(ENV, "").strip()
    return path or None


def evidence_path(key, *rel):
    """<KEY of the opted-in machine config>/<rel...>, or None when the opt-in is off or the key is missing."""
    path = machine_config_path()
    if not path:
        return None
    lc = _loader()
    try:
        return os.path.join(lc.load(path).get(key), *rel)
    except lc.LocalConfigError:
        return None


def skip_reason(what):
    return "set %s to this machine's drivedit.local.env to compare with %s (DD-22 opt-in)" % (ENV, what)


SCRATCH_ENV = "DC_TEST_E_SCRATCH"


def _identities(path):
    """(st_dev, st_ino) of path and of each existing ancestor."""
    out = set()
    p = path
    while True:
        try:
            st = os.stat(p)
            out.add((st.st_dev, st.st_ino))
        except OSError:
            pass
        parent = os.path.dirname(p)
        if parent == p:
            return out
        p = parent


def _overlap(a, b):
    """a and b are one directory or one lies inside the other: letter case ignored (drvfs) and filesystem identity
    compared, so a case alias or a link counts too."""
    la, lb = a.lower().rstrip("/") + "/", b.lower().rstrip("/") + "/"
    if la.startswith(lb) or lb.startswith(la):
        return True
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return (sb.st_dev, sb.st_ino) in _identities(a) or (sa.st_dev, sa.st_ino) in _identities(b)


def data_drive_scratch():
    """(scratch dir, None) when DC_TEST_E_SCRATCH and DD_TEST_LOCAL_CONFIG are both set and the scratch directory is
    safe to create test directories in, else (None, why). Everything is checked before anything is created:
      * DC_TEST_E_SCRATCH is an existing directory, not a symlink; its resolved path is what is used;
      * the write boundary of tools/local_config.py, with the machine's config, allows writing there (check_write:
        mountinfo, device, no Windows/System32 at the volume root), and it lies on that config's verified data
        drive itself, not in RAM or on any other mount;
      * it lies apart from the production DATA_ROOT (neither inside the other; letter case and identity compared)."""
    scratch = os.environ.get(SCRATCH_ENV, "").strip()
    machine = machine_config_path()
    if not (scratch and machine):
        return None, ("set %s (a directory on the data drive, outside DATA_ROOT) and %s to run this test"
                      % (SCRATCH_ENV, ENV))
    lc = _loader()
    try:
        cfg = lc.load(machine)
        data_root = cfg.get("DATA_ROOT")
    except lc.LocalConfigError as exc:
        return None, "%s: %s" % (ENV, exc)
    if os.path.islink(scratch) or not os.path.isdir(scratch):
        return None, "%s=%s is not an existing directory (or is a symlink)" % (SCRATCH_ENV, scratch)
    with lc.override(cfg):
        try:
            resolved = lc.check_write(os.path.realpath(scratch))
            mounts = lc.read_mountinfo()
        except lc.WriteRefused as exc:
            return None, "%s=%s: %s" % (SCRATCH_ENV, scratch, exc)
        vol, why = lc.data_volume(mounts)
    if vol is None or lc.mount_of(resolved, mounts) is not vol:
        return None, "%s=%s is not on the data drive of %s (%s)" % (SCRATCH_ENV, scratch, machine,
                                                                     why or "it is in RAM or on another mount")
    if _overlap(resolved, os.path.realpath(lc.mounted_path(data_root))):
        return None, "%s=%s overlaps the production DATA_ROOT %s" % (SCRATCH_ENV, scratch, data_root)
    return resolved, None
