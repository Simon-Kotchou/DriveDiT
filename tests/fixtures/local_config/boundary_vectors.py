"""The write boundary's test vectors and the fake world they run in (DD-22 round 2). Stdlib only, Python 3.10+.

One table drives every entry point of the boundary: tools/local_config.py check_write (the tools, pinlib and the
shell scripts' check-write) and drivedit.local_paths.check_write (the package). Each vector names a destination, the
caller's roots, and the outcome under every config state.

The world is a directory under /dev/shm with a fake mountinfo; nothing outside it is ever created, and the boundary
only reads (a vector checks, it never writes). In the world, <base>/mnt stands for the drive mounts (MNT_ROOT) and
<base>/shm for RAM (SCRATCH_MOUNT):

  x  a drvfs mount of drive X: (the whole drive), no Windows/System32: the data drive of the real-like config
  z  a drvfs mount of drive Z: holding Windows/System32: the Windows system drive
  w  a drvfs mount of drive W: holding windows/SYSTEM32 (other letter case): a second system-marked volume
  v  a drvfs mount at <mnt>/v whose source is drive U: (the name of a mount is not its drive)
  u  a drvfs mount of drive U: that mounts its \\sub only (not the whole drive)
  y  a plain directory, mounted nowhere: an unmounted drive (on WSL a directory of the Linux root filesystem)
plus, below x: <x>/DataRoot/nested (another filesystem mounted inside the data drive) and <x>/DataRoot/otherdev (a
mountinfo entry whose device is not the one the path has).

Config states (STATES): real_like, placeholder (the synthetic config's drives, mounted nowhere), missing, malformed,
marked_data (DATA_ROOT on a System32-marked volume), swapped (DATA_ROOT under the system drive and SYSTEM_DRIVE naming
the data drive), wrong_system_drive (only SYSTEM_DRIVE wrong: it is advisory), unmounted_data, wrong_letter, partial.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER = os.path.join(HERE, "placeholder.local.env")

_REST = ("REALIGN_ROOT=/mnt/x/PlanRoot\nCODE_ROOT=/mnt/x/CodeRoot\n"
         "GAME_INSTALL=/mnt/y/Game Library/steamapps/common/Arma Reforger\n"
         "TOOLS_INSTALL=/mnt/z/Synthetic Steam (x86)/steamapps/common/Arma Reforger Tools\n"
         "WSL_DISTRO=Placeholder-Distro\n")


def _mnt(letter: str) -> str:
    """/mnt/<letter>, assembled: drives other than the placeholders X, Y and Z appear only in the fake world."""
    return "/mnt/" + letter


def _cfg(data_root: str, system: str) -> str:
    return "DATA_ROOT=%s\nSYSTEM_DRIVE=%s\n%s" % (data_root, system, _REST)


# state -> (config text or None for no config, the world's drive-mount dir: "mnt" or "nomnt")
STATES = {
    "real_like": (_cfg("/mnt/x/DataRoot", "/mnt/z"), "mnt"),
    "placeholder": ("@placeholder", "nomnt"),
    "missing": (None, "mnt"),
    "malformed": (_cfg("/mnt/x/DataRoot", "/mnt/z") + "DATA_ROOT=/mnt/x/DataRoot\n", "mnt"),
    "marked_data": (_cfg(_mnt("w") + "/DataRoot", "/mnt/z"), "mnt"),
    "swapped": (_cfg("/mnt/z/DataRoot", "/mnt/x"), "mnt"),
    "wrong_system_drive": (_cfg("/mnt/x/DataRoot", "/mnt/x"), "mnt"),
    "unmounted_data": (_cfg("/mnt/y/DataRoot", "/mnt/z"), "mnt"),
    "wrong_letter": (_cfg(_mnt("v") + "/DataRoot", "/mnt/z"), "mnt"),
    "partial": (_cfg(_mnt("u") + "/DataRoot", "/mnt/z"), "mnt"),
}
DATA_OK = ("real_like", "wrong_system_drive")   # the states with a verified data drive (x)
ALL = tuple(STATES)
NONE = ()

# (id, destination, roots or None, states in which the write is allowed, a fragment every refusal holds or None).
# {B} is the world's base, {M} its drive-mount dir (<base>/mnt).
VECTORS = (
    ("ram", "{B}/shm/out.json", None, ALL, None),
    ("ram_new_dirs", "{B}/shm/a/b/c.json", None, ALL, None),
    ("data_root", "{M}/x/DataRoot/out.json", None, DATA_OK, None),
    ("data_drive_elsewhere", "{M}/x/elsewhere/new/out.json", None, DATA_OK, None),
    ("data_drive_windows_form", "X:\\DataRoot\\out.json", None, NONE, None),   # the real /mnt/x: not in the world
    ("system_drive", "{M}/z/out.json", None, NONE, "Windows system drive"),
    ("system_drive_data_root", "{M}/z/DataRoot/out.json", None, NONE, "Windows system drive"),
    ("marked_other_case", "{M}/w/DataRoot/out.json", None, NONE, "Windows system drive"),
    ("unmounted_drive", "{M}/y/DataRoot/out.json", None, NONE, "refusing"),
    ("wrong_letter_mount", "{M}/v/DataRoot/out.json", None, NONE, "refusing"),
    ("partial_mount", "{M}/u/DataRoot/out.json", None, NONE, "refusing"),
    ("other_tmpfs", "{B}/elsewhere/out.json", None, NONE, "refusing"),
    ("tmp", "/tmp/dd22-boundary-vector-never-written/out.json", None, NONE, "refusing"),
    ("rootfs", "/var/dd22-boundary-vector-never-written/out.json", None, NONE, "refusing"),
    ("dotdot_escape", "{B}/shm/../mnt/z/out.json", None, NONE, "'..'"),
    ("dotdot_in_ram", "{B}/shm/a/../b.json", None, NONE, "'..'"),
    ("relative_dotdot", "../out.json", None, NONE, "'..'"),
    ("symlink_leaf", "{B}/shm/leaf_link", None, NONE, "symlink"),
    ("symlink_leaf_to_system", "{B}/shm/leaf_to_system", None, NONE, "symlink"),
    ("symlink_dir_to_system", "{B}/shm/dir_to_system/out.json", None, NONE, "Windows system drive"),
    ("symlink_dir_to_tmp", "{B}/shm/dir_to_tmp/out.json", None, NONE, "refusing"),
    ("symlink_dir_in_ram", "{B}/shm/dir_in_ram/out.json", None, ALL, None),
    ("dangling_dir", "{B}/shm/dangling/out.json", None, NONE, "dangling"),
    ("nested_mount", "{M}/x/DataRoot/nested/out.json", None, NONE, "refusing"),
    ("device_mismatch", "{M}/x/DataRoot/otherdev/out.json", None, NONE, "device"),
    ("widen_to_tmp", "/tmp/dd22-boundary-vector-never-written/out.json", ["/tmp"], NONE, "refusing"),
    ("widen_to_root", "{M}/z/out.json", ["/"], NONE, "Windows system drive"),
    ("widen_to_unmounted", "{M}/y/out.json", ["{M}/y"], NONE, "refusing"),
    ("widen_to_other_tmpfs", "{B}/elsewhere/out.json", ["{B}"], NONE, "refusing"),
    ("narrow_ram_inside", "{B}/shm/n/out.json", ["{B}/shm/n"], ALL, None),
    ("narrow_ram_outside", "{B}/shm/out.json", ["{B}/shm/n"], NONE, "outside"),
    ("narrow_data_inside", "{M}/x/DataRoot/out.json", ["{M}/x/DataRoot"], DATA_OK, None),
    ("narrow_data_outside", "{M}/x/elsewhere/out.json", ["{M}/x/DataRoot"], NONE, None),
    ("no_roots", "{B}/shm/out.json", [], NONE, "outside"),
    ("empty_path", "", None, NONE, "not a path"),
)

# (child name, allowed): names derived from inputs (CSV names, session names, header files) under a checked dir.
CHILD_NAMES = (
    ("out.csv", True),
    ("run_header.txt", True),
    (".hidden", True),
    ("/tmp/abs.csv", False),
    ("/abs.csv", False),
    ("../up.csv", False),
    ("..", False),
    (".", False),
    ("a/b.csv", False),
    ("a\\b.csv", False),
    ("C:x.csv", False),
    ("Z:\\x.csv", False),
    ("", False),
)


def _mi_escape(path: str) -> str:
    return path.replace("\\", "\\134").replace(" ", "\\040").replace("\t", "\\011").replace("\n", "\\012")


def _dev(st_dev: int) -> str:
    return "%d:%d" % (os.major(st_dev), os.minor(st_dev))


class World:
    """The fake filesystem and mountinfo under base; use(lc, state) points a loader module at it."""

    def __init__(self, base: str):
        self.base = base
        self.mnt = os.path.join(base, "mnt")
        self.shm = os.path.join(base, "shm")
        for rel in ("shm", "shm/real", "elsewhere", "nomnt", "mnt/x/DataRoot/nested", "mnt/x/DataRoot/otherdev",
                    "mnt/x/elsewhere", "mnt/z/Windows/System32", "mnt/z/DataRoot", "mnt/w/windows/SYSTEM32",
                    "mnt/w/DataRoot", "mnt/v/DataRoot", "mnt/u/DataRoot", "mnt/y/DataRoot", "configs"):
            os.makedirs(os.path.join(base, rel), exist_ok=True)
        with open(os.path.join(self.shm, "real", "target"), "w") as fh:
            fh.write("target\n")
        links = {"leaf_link": "real/target", "leaf_to_system": self.mnt + "/z/never.json",
                 "dir_to_system": self.mnt + "/z", "dir_to_tmp": "/tmp", "dir_in_ram": "real",
                 "dangling": "real/absent_dir"}
        for name, target in links.items():
            os.symlink(target, os.path.join(self.shm, name))
        here = _dev(os.stat(base).st_dev)
        root = _dev(os.stat("/").st_dev)
        shm = _dev(os.stat("/dev/shm").st_dev)

        def line(n, parent, dev, mroot, point, fstype, source, opts="rw"):
            return "%d %d %s %s %s rw,noatime - %s %s %s\n" % (n, parent, dev, _mi_escape(mroot), _mi_escape(point),
                                                              fstype, _mi_escape(source), _mi_escape(opts))

        def src(letter):   # the mountinfo source of a drvfs mount: the drive's root
            return letter + ":" + "\\"

        def drive(letter):
            return "rw,dirsync,aname=drvfs;path=%s;uid=1000;gid=1000" % src(letter)

        self.mountinfo = "".join((
            line(1, 0, root, "/", "/", "ext4", "/dev/root"),
            line(2, 1, shm, "/", "/dev/shm", "tmpfs", "none"),
            line(3, 2, here, "/", self.shm, "tmpfs", "none"),
            line(4, 2, here, "/", self.mnt + "/x", "9p", src("X"), drive("X")),
            line(5, 2, here, "/", self.mnt + "/z", "9p", src("Z"), drive("Z")),
            line(6, 2, here, "/", self.mnt + "/w", "9p", src("W"), drive("W")),
            line(7, 2, here, "/", self.mnt + "/v", "9p", src("U"), drive("U")),
            line(8, 2, here, "/sub", self.mnt + "/u", "9p", src("U"), drive("U")),
            line(9, 4, here, "/", self.mnt + "/x/DataRoot/nested", "tmpfs", "none"),
            line(10, 4, "0:999", "/", self.mnt + "/x/DataRoot/otherdev", "9p", src("X"), drive("X")),
        ))
        self.mountinfo_path = os.path.join(base, "mountinfo")
        with open(self.mountinfo_path, "w") as fh:
            fh.write(self.mountinfo)

    def path(self, template: str) -> str:
        return template.replace("{B}", self.base).replace("{M}", self.mnt)

    def config_path(self, state: str):
        text, _ = STATES[state]
        if text is None:
            return None
        if text == "@placeholder":
            return PLACEHOLDER
        path = os.path.join(self.base, "configs", state + ".local.env")
        with open(path, "w") as fh:
            fh.write(text)
        return path

    @contextlib.contextmanager
    def use(self, lc, state: str = "real_like"):
        """lc (a loaded tools/local_config.py) sees this world and the config of state, for the block."""
        saved = {k: getattr(lc, k) for k in ("MOUNTINFO", "SCRATCH_MOUNT", "MNT_ROOT")}
        env = os.environ.get(lc.ENV_VAR)
        cfg_path = self.config_path(state)
        lc.MOUNTINFO = self.mountinfo_path
        lc.SCRATCH_MOUNT = self.shm
        lc.MNT_ROOT = os.path.join(self.base, STATES[state][1])
        lc._MARKER_CACHE.clear()
        os.environ[lc.ENV_VAR] = cfg_path or os.path.join(self.base, "configs", "no-such.local.env")
        try:
            yield self
        finally:
            for k, v in saved.items():
                setattr(lc, k, v)
            lc._MARKER_CACHE.clear()
            if env is None:
                os.environ.pop(lc.ENV_VAR, None)
            else:
                os.environ[lc.ENV_VAR] = env


@contextlib.contextmanager
def world(prefix: str = "dd22_boundary_"):
    """A fresh World in a directory under /dev/shm, removed afterwards."""
    base = os.path.realpath(tempfile.mkdtemp(prefix=prefix, dir="/dev/shm"))
    try:
        yield World(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)


def system_volume(lc, base: str, name: str = "fake_system_drive"):
    """A directory under base that the boundary of lc (patched for the caller's block, see patched_system_volume)
    takes for the Windows system drive: a drvfs mount whose root holds Windows/System32."""
    root = os.path.join(base, name)
    os.makedirs(os.path.join(root, "Windows", "System32"), exist_ok=True)
    return root


@contextlib.contextmanager
def patched_system_volume(lc, base: str, name: str = "fake_system_drive"):
    """lc's boundary sees the real mountinfo plus one drvfs mount of drive Z: at <base>/<name> whose root holds
    Windows/System32; yields that root (its subdirectories do not exist yet). For tests that check a writer refuses the
    system drive without ever being given a real one."""
    root = system_volume(lc, base, name)
    with open(lc.MOUNTINFO if os.path.isfile(lc.MOUNTINFO) else "/proc/self/mountinfo") as fh:
        real = fh.read()
    fake = os.path.join(base, name + ".mountinfo")
    with open(fake, "w") as fh:
        fh.write(real + "9999 1 %s / %s rw,noatime - 9p Z:\\134 rw,aname=drvfs;path=Z:\\\n"
                 % (_dev(os.stat(base).st_dev), _mi_escape(root)))
    saved = lc.MOUNTINFO
    lc.MOUNTINFO = fake
    lc._MARKER_CACHE.clear()
    try:
        yield root
    finally:
        lc.MOUNTINFO = saved
        lc._MARKER_CACHE.clear()
