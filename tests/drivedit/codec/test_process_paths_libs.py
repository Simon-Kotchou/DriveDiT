"""The defaults of ``drivedit.process_paths`` are the installed libraries' own derivations (DD-23): given the parents
``prepare_process`` sets, huggingface_hub, torch, diffusers and Triton (when installed) arrive at the same directories,
so setting a child explicitly changes nothing where its parent already pointed it. Read in a subprocess, which only
computes paths."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from drivedit.process_paths import PROCESS_PATHS, resolve
from tests.drivedit.boundary import world

PROBE = r"""
import json, os
out = {}
import huggingface_hub.constants as c
out.update(HF_HOME=c.HF_HOME, HF_HUB_CACHE=c.HF_HUB_CACHE, HF_XET_CACHE=c.HF_XET_CACHE)
out["HF_ASSETS_CACHE"] = c.HF_ASSETS_CACHE
import torch.hub, torch._appdirs
out["TORCH_HOME"] = os.path.dirname(torch.hub.get_dir())
out["TORCH_EXTENSIONS_DIR"] = torch._appdirs.user_cache_dir(appname="torch_extensions")
try:
    from diffusers.utils.constants import HF_MODULES_CACHE
    out["HF_MODULES_CACHE"] = HF_MODULES_CACHE
except ImportError:
    pass
try:
    import triton.knobs
    out["TRITON_CACHE_DIR"] = triton.knobs.cache.dir
    out["TRITON_DUMP_DIR"] = triton.knobs.cache.dump_dir
    out["TRITON_OVERRIDE_DIR"] = triton.knobs.cache.override_dir
except ImportError:
    pass
print(json.dumps(out))
"""


def test_library_defaults_match_the_table(tmp_path: Path) -> None:
    with world() as w:
        ours = resolve({})
        cache = f"{w.mnt}/x/DataRoot/cache"
        assert ours["XDG_CACHE_HOME"] == cache
    # Only the parents the libraries read: the roots and TRITON_HOME (Triton's default follows HOME, not XDG).
    env = {k: v for k, v in os.environ.items() if k not in {e.name for e in PROCESS_PATHS}}
    env |= {"XDG_CACHE_HOME": ours["XDG_CACHE_HOME"], "TMPDIR": str(tmp_path), "TRITON_HOME": ours["TRITON_HOME"]}
    result = subprocess.run(
        [sys.executable, "-I", "-c", PROBE], cwd=tmp_path, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    theirs: dict[str, str] = json.loads(result.stdout)
    children = {e.name: os.path.join(ours[e.base], e.child) for e in PROCESS_PATHS if e.base in ours}
    for name, value in theirs.items():
        assert value == ours.get(name, children.get(name)), name
    assert {"HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR"} <= set(theirs)
