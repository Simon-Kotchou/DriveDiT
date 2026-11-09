"""The frozen DINOv3 ViT-B/16 that CADRE encodes with, built from a verified local checkout (DD-19).

CADRE's checkpoints carry the DINOv3 weights but not its model code, which is Meta's (DINOv3 License, not MIT), so it
is not vendored here. It is imported from a local checkout of ``facebookresearch/dinov3`` at commit
``DINOV3_COMMIT``, the commit the CADRE checkpoints were trained with:

- **Where.** The ``directory`` keyword, else the ``DRIVEDIT_DINOV3_DIR`` environment variable, else
  ``default_dinov3_dir()``: ``<CODE_ROOT>/third_party/dinov3`` of the local config (DD-22, ``drivedit.local_paths``).
  Fetch it once with the command ``fetch_command()`` prints; nothing here touches the network.
- **What runs.** Only the 17 files of ``DINOV3_SHA256``: the ``dinov3.hub.backbones`` entry point and the ViT code it
  builds. Every file's sha256 is checked before anything is imported, and each file is hashed again as it is executed
  (the bytes checked are the bytes run). A ``dinov3`` module outside the manifest cannot be imported: the import hook
  refuses it, and the packages get an empty ``__path__``, so no other finder can load one later either.
- **Why not torch.hub.** ``torch.hub.load(..., source="local")`` would import the repository's ``hubconf.py``, which
  re-exports the same ``dinov3_vitb16`` but first imports DINOv3's classifier, detector, segmentor, depther and text
  code: 55 files instead of 17, plus torchvision (measured on the checkout, 2026-11-05).
- **Names.** DINOv3 imports itself by its absolute name ``dinov3``, so its modules live in ``sys.modules`` under that
  name. A ``dinov3`` module that this loader did not load from the same checkout raises instead of being reused.

The model is built with ``pretrained=False``, so no weights download: the CADRE checkpoint loads them.

**Licence.** The DINOv3 License (``LICENSE.md`` of the checkout). Its section v forbids "end uses prohibited by Trade
Controls, including those related to military or warfare purposes" and ITAR-controlled activities; the user decides
whether that admits this project (MODELING_PLAN U1).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.util
import os
import sys
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from drivedit import local_paths

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from importlib.machinery import ModuleSpec
    from types import ModuleType

    import torch

__all__ = [
    "DINOV3_COMMIT",
    "DINOV3_DIR_ENV",
    "DINOV3_DIR_REL",
    "DINOV3_SHA256",
    "DinoV3SourceError",
    "default_dinov3_dir",
    "dinov3_dir",
    "fetch_command",
    "load_dinov3_vitb16",
    "verify_checkout",
]

DINOV3_COMMIT: Final = "1e358a23bc05795715534db39fbca26fc442d8c8"
DINOV3_REPO: Final = "https://github.com/facebookresearch/dinov3"
DINOV3_DIR_ENV: Final = "DRIVEDIT_DINOV3_DIR"
DINOV3_DIR_REL: Final = "third_party/dinov3"  # the default checkout, under CODE_ROOT of the local config (DD-22)
PACKAGE: Final = "dinov3"
# sha256 of every file that `from dinov3.hub.backbones import dinov3_vitb16; dinov3_vitb16(pretrained=False)` imports
# at DINOV3_COMMIT (dinov3/hub is a namespace package: it has no __init__.py). A forward pass imports nothing more.
DINOV3_SHA256: Final[Mapping[str, str]] = MappingProxyType(
    {
        "dinov3/__init__.py": "cfcd49b9d41705ef9bd6ce61bccc38e8ab8fa52fa365de48c10a7d7e1710b631",
        "dinov3/hub/backbones.py": "1b4a111e028e7a414c40f8b4bb84df6aac9c509bcc62aa6979ead92e8a81daae",
        "dinov3/hub/utils.py": "292698a622db0dcf393ed1f987688b25c55822634a48bdd7d883760c5b1f65a6",
        "dinov3/layers/__init__.py": "0707ef3e9c20aff594c3817f64a2c8ff34a21bd3b55e7d03471969fd314dde20",
        "dinov3/layers/attention.py": "dfc21def6ba17d00cba18ea23a8726e7f1dd5a0dc8e43510852c999a173d37a8",
        "dinov3/layers/block.py": "809a3615ec93019042ab440f31c21d5242186d9e503c67ac0f42a324ceae1950",
        "dinov3/layers/ffn_layers.py": "fa2caadb52ff43a7e6888966a00ee07842fed9fd05c68305962cc1577d221fbe",
        "dinov3/layers/fp8_linear.py": "f5b4d46e8c159e1830bc99968baf9a2e8070f0c46d0dda235eac46cc11c34f47",
        "dinov3/layers/layer_scale.py": "f9a10ee763a62136f9e814bfe2f2209c2cb501dfdc5573f858ab01ba9a69c23f",
        "dinov3/layers/patch_embed.py": "c592c7262779c8e1789ed502aee81bdd55eba4607065f8d04ae95eea683d0ef1",
        "dinov3/layers/rms_norm.py": "8509ffc53f0bbe9e353a8b9e57ca2396795640b4ecc85140c27b17127c971869",
        "dinov3/layers/rope_position_encoding.py": "af8ea35acb76d18ad4d53b844d5057bf01472a848375c51d70583a87f044efd1",
        "dinov3/models/__init__.py": "4479dfb19068e2599aaf35c6c698a2cf5b1ecb0b639c6b7afdb647570b7dd987",
        "dinov3/models/vision_transformer.py": "1af05405e19bc42188f80bfcb204b87ed1ec72af998e0bbd4658a85d7fae93f2",
        "dinov3/utils/__init__.py": "83e6ec1f0cea862f18ae111560e2eadf2fff1873e3fdd7d358dee8d32d0ade8d",
        "dinov3/utils/dtype.py": "0c53f5cc155ea7e9899bc611493193fef9b985f23cb7d990401e22c363ac4419",
        "dinov3/utils/utils.py": "d6fe7fe621b2ae3db89e3af588fbee0c78afe2130f4548dff87c94cbd734fe40",
    }
)


class DinoV3SourceError(RuntimeError):
    """The DINOv3 checkout is missing or differs from the manifest, or a foreign ``dinov3`` module is loaded."""


def default_dinov3_dir() -> Path:
    """``<CODE_ROOT>/third_party/dinov3`` of the local config (DD-22); DinoV3SourceError when no config names it."""
    try:
        return Path(local_paths.value("CODE_ROOT")) / DINOV3_DIR_REL
    except local_paths.LocalPathsError as exc:
        raise DinoV3SourceError(f"no DINOv3 checkout is configured: set {DINOV3_DIR_ENV}, or {exc}") from exc


def dinov3_dir(directory: Path | str | None = None) -> Path:
    """The checkout to use: ``directory``, else ``$DRIVEDIT_DINOV3_DIR``, else ``default_dinov3_dir()``."""
    if directory is not None:
        return Path(directory)
    value = os.environ.get(DINOV3_DIR_ENV)
    return Path(value) if value else default_dinov3_dir()


def fetch_command(directory: Path) -> str:
    """The shell command that creates the pinned checkout at ``directory`` (run it by hand; this module never does)."""
    return (
        f"git init {directory} && git -C {directory} fetch --depth 1 {DINOV3_REPO} {DINOV3_COMMIT} "
        f"&& git -C {directory} checkout FETCH_HEAD"
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_checkout(directory: Path) -> Path:
    """The resolved ``directory`` after checking every manifest file; ``DinoV3SourceError`` names the fetch command."""
    root = directory.resolve()
    for relative, expected in DINOV3_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise DinoV3SourceError(
                f"{path} is missing: {directory} is not a DINOv3 checkout at {DINOV3_COMMIT}. Fetch it with: "
                f"{fetch_command(directory)} (or point {DINOV3_DIR_ENV} at one)"
            )
        if _digest(path.read_bytes()) != expected:
            raise DinoV3SourceError(f"{path} differs from DINOv3 commit {DINOV3_COMMIT} (sha256 mismatch)")
    return root


class _VerifiedLoader(importlib.abc.Loader):
    """Executes one manifest file, hashing the exact bytes it runs; ``relative is None`` is a namespace package."""

    def __init__(self, root: Path, relative: str | None) -> None:
        self.root = root
        self.relative = relative

    def create_module(self, spec: ModuleSpec) -> None:
        return None  # the default module object

    def exec_module(self, module: ModuleType) -> None:
        if self.relative is None:
            return
        path = self.root / self.relative
        source = path.read_bytes()
        if _digest(source) != DINOV3_SHA256[self.relative]:
            raise DinoV3SourceError(f"{path} changed after verification (sha256 mismatch)")
        exec(compile(source, str(path), "exec"), module.__dict__)  # the hash-verified bytes of a manifest file


class _VerifiedFinder(importlib.abc.MetaPathFinder):
    """Resolves ``dinov3`` and its submodules to manifest files of one checkout, and refuses every other name."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if fullname != PACKAGE and not fullname.startswith(PACKAGE + "."):
            return None
        base = fullname.replace(".", "/")
        relative: str | None
        if f"{base}/__init__.py" in DINOV3_SHA256:
            relative, package = f"{base}/__init__.py", True
        elif f"{base}.py" in DINOV3_SHA256:
            relative, package = f"{base}.py", False
        elif any(name.startswith(base + "/") for name in DINOV3_SHA256):
            relative, package = None, True  # a namespace package, such as dinov3.hub
        else:
            raise ImportError(f"{fullname} is not in the verified DINOv3 manifest", name=fullname)
        origin = str(self.root / relative) if relative is not None else None
        spec = importlib.util.spec_from_loader(
            fullname, _VerifiedLoader(self.root, relative), origin=origin, is_package=package
        )
        if spec is None:
            raise ImportError(f"cannot build a module spec for {fullname}", name=fullname)
        spec.has_location = relative is not None
        # is_package gives an empty submodule_search_locations: children resolve through this finder only.
        return spec


def _check_loaded(root: Path) -> None:
    """Raise if any loaded ``dinov3`` module came from somewhere other than this loader on ``root``."""
    for name, module in list(sys.modules.items()):
        if name != PACKAGE and not name.startswith(PACKAGE + "."):
            continue
        loader = getattr(getattr(module, "__spec__", None), "loader", None)
        if not (isinstance(loader, _VerifiedLoader) and loader.root == root):
            raise DinoV3SourceError(
                f"a module named {name!r} that was not loaded from the verified checkout {root} is already imported"
            )


def load_dinov3_vitb16(directory: Path | str | None = None) -> torch.nn.Module:
    """DINOv3 ViT-B/16 without weights, frozen, built on the default device from the verified checkout."""
    root = dinov3_dir(directory)
    _check_loaded(root.resolve())  # before reading files, so a foreign module is reported as such
    root = verify_checkout(root)
    finder = _VerifiedFinder(root)
    sys.meta_path.insert(0, finder)
    try:
        backbones = importlib.import_module(f"{PACKAGE}.hub.backbones")
        model: torch.nn.Module = backbones.dinov3_vitb16(pretrained=False)  # imports the ViT code lazily
    finally:
        sys.meta_path.remove(finder)
    _check_loaded(root)
    return model.requires_grad_(False)
