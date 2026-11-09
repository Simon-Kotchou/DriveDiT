"""Scratch directories for the tests (DD-22), validated before anything is created. Stdlib only, Python 3.10+.

Every test scratch lives in RAM. ram_base() takes the directory an override variable names (DD_PIN_TEST_TMP,
DEPLOY_ADDON_TEST_TMP) when one is given and set, else $TMPDIR when it lies under /dev/shm, else /dev/shm, and checks
it before use: the path is no symlink, its resolved form lies under /dev/shm, is an existing writable directory and
lies on a tmpfs according to /proc/self/mountinfo. An override that fails is refused, never replaced by another place.
Nothing here falls back to tempfile.gettempdir() or /tmp.

boundary_base() also needs the write boundary of tools/local_config.py, reading the real mountinfo, to allow writing
there: the tests of the real boundary (the tools write through it) use it. Where the boundary refuses RAM (a sandbox
may hide the RAM mount, so that mountinfo names another device than the directory has; the boundary then refuses,
fail-closed), those tests skip with that reason. The policy tests, which use a fake mountinfo
(boundary_vectors.py), take ram_base() and do not skip.

    spec = importlib.util.spec_from_file_location("dd22_scratch", "<repo>/tests/fixtures/local_config/scratch.py")
"""
import importlib.util
import os
import tempfile
import unittest

RAM = "/dev/shm"
REAL_MOUNTINFO = "/proc/self/mountinfo"
HERE = os.path.dirname(os.path.abspath(__file__))
LOADER = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "tools", "local_config.py")
_LC = []


def _lc():
    """tools/local_config.py as a module of its own (so a test that patches the shared one changes nothing here)."""
    if not _LC:
        spec = importlib.util.spec_from_file_location("dd22_scratch_local_config", LOADER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _LC.append(mod)
    return _LC[0]


def ram_problem(path):
    """Why path cannot be a RAM scratch base, or None."""
    if not path:
        return "no path"
    if os.path.islink(path):
        return "%s is a symlink" % path
    real = os.path.realpath(path)
    if not (real == RAM or real.startswith(RAM + "/")):
        return "%s resolves to %s, which is not under %s" % (path, real, RAM)
    if not os.path.isdir(real):
        return "%s is not an existing directory" % real
    if not os.access(real, os.W_OK):
        return "%s is not writable" % real
    lc = _lc()
    try:
        with open(REAL_MOUNTINFO, encoding="utf-8", errors="replace") as fh:
            mounts = lc.parse_mountinfo(fh.read())
    except OSError as exc:
        return "cannot read %s: %s" % (REAL_MOUNTINFO, exc)
    m = lc.mount_of(real, mounts)
    if m is None or m["fstype"] != "tmpfs":
        return "%s is not on a tmpfs (mountinfo: %s)" % (real, "no mount" if m is None else
                                                       "%s, %s" % (m["mount_point"], m["fstype"]))
    return None


def ram_base(override=None):
    """The validated RAM scratch base (see the module docstring). unittest.SkipTest naming the reason when there is
    none; an override variable that is set must pass, or the tests skip (it is never replaced by another place)."""
    if override and os.environ.get(override, "").strip():
        path = os.environ[override].strip()
        why = ram_problem(path)
        if why:
            raise unittest.SkipTest("$%s is not a usable RAM scratch directory: %s (it must be a directory under %s; "
                                    "nothing was created)" % (override, why, RAM))
        return os.path.realpath(path)
    reasons = []
    tmpdir = os.environ.get("TMPDIR", "").strip()
    cands = ([tmpdir] if tmpdir and os.path.realpath(tmpdir).startswith(RAM + "/") else []) + [RAM]
    for cand in cands:
        why = ram_problem(cand)
        if why is None:
            return os.path.realpath(cand)
        reasons.append(why)
    raise unittest.SkipTest("no usable RAM scratch directory (%s); the tests never use the system temp dir"
                            % "; ".join(reasons))


def boundary_base(override=None):
    """ram_base(), which the real write boundary must also allow (tests of the real boundary); SkipTest otherwise."""
    base = ram_base(override)
    why = _lc().write_refusal(os.path.join(base, "__dd22_scratch_probe__"))
    if why:
        raise unittest.SkipTest("the write boundary refuses RAM scratch here, fail-closed (%s); the tests that use "
                                "the real boundary skip" % why)
    return base


def mkdtemp(prefix, real_boundary=True, override=None):
    """A new directory under boundary_base() (real_boundary) or ram_base(), named prefix + random."""
    base = boundary_base(override) if real_boundary else ram_base(override)
    return tempfile.mkdtemp(prefix=prefix, dir=base)
