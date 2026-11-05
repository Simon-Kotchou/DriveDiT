"""CADRE codecs (Kyutai, arXiv 2610.30129): ``cadre-1x16x16`` and ``cadre-1x48x48``.

CADRE is a family of causal video autoencoders for interactive world models. Its encoder is a frozen DINOv3 ViT-B/16
(features of layers 2 and 4-11 averaged, plus the last) followed by one trained strided convolution down to 64
channels; its decoder is a ~475M space-time transformer that samples the clip back by flow matching in 1-2 steps,
causally in time. Both cells used here are tau=1 (one latent per frame), trained at 288x512 and 20 fps:

| codec id | latent at 288x512 | tokens per frame (delta-2026-11 §2) |
|---|---|---|
| ``cadre-1x16x16`` | ``[64, 18, 32]`` | 144 at DiT patch 2 |
| ``cadre-1x48x48`` | ``[64, 6, 11]`` (the 18x32 DINO grid zero-padded to 18x33, stride 3) | 66 at patch 1 |

The tau=2 cells are not used: at 5 Hz one tau=2 latent would pair frames 400 ms apart, outside the 20 fps training
distribution (delta-2026-11 §2).

**Latent space.** ``encode`` returns CADRE's standardised latents, ``(z - latent_mean) / latent_std`` with the scalar
statistics of the checkpoint's ``config.json`` (``CADRE.normalize``, which the model card says a world model uses).

**Decoding.** ``decode`` samples every latent as its own one-frame clip, with ``decode_steps`` flow steps and noise from
a generator seeded with ``decode_seed`` on every call, so one call is reproducible (the noise of a frame depends on its
position in the batch). ``decode_clip`` instead decodes consecutive latents as one causal clip, so the decoder's
temporal attention and its causal temporal convolutions see the history. The decoder was trained on 20 fps clips; at
our 5 Hz each step carries about 4x the motion, so clip decoding at 5 Hz is unverified until measured
(``tools/codec/measure_codecs.py``; MODELING_PLAN §3 and §9).

**Loading.** CADRE is not a package: its ``pyproject.toml`` has no build system (``package = false``), pins
``torch==2.8.0`` and ``requires-python >= 3.12``, and its modules are top-level (``model_autoencoder``, ``util``). So it
is loaded from a source checkout at the pinned commit, named by the ``DRIVEDIT_CADRE_SRC`` environment variable::

    git init cadre && cd cadre
    git remote add origin https://github.com/kyutai-labs/cadre-autoencoders-and-world-models
    git fetch --depth 1 origin 3b07314239bf73fe952a98aab78ae03e1937d8bf && git checkout FETCH_HEAD

The loader checks the sha256 of every CADRE file it imports against ``CADRE_SOURCE_SHA256`` and refuses a mismatch.
CADRE builds its DINOv3 ViT-B/16 from ``torch.hub`` (``facebookresearch/dinov3`` at commit ``1e358a23``, executed with
``trust_repo=True``; it downloads to ``torch.hub.get_dir()``, i.e. ``$TORCH_HOME/hub``), without weights: the CADRE
checkpoint carries the DINOv3 weights. The checkpoint downloads from the Hub at the pinned revision of ``CODECS``.

**Licences** (flag for the user; not decided here):

- CADRE weights: CC-BY-NC-4.0 (non-commercial use only).
- The frozen DINOv3 ViT-B/16 weights inside them, and the DINOv3 model code from torch.hub: the DINOv3 License, whose
  section v forbids "end uses prohibited by Trade Controls, including those related to military or warfare purposes"
  and ITAR-controlled activities. Arma Reforger is a military-themed game; whether a driving world model trained on it
  falls under that clause needs the user's read (MODELING_PLAN U1).
- CADRE code: MIT.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import torch

from drivedit.codec.base import CODECS, Codec, FrozenCodec, fetch_weights, unit_to_images

if TYPE_CHECKING:
    from types import ModuleType

    from drivedit.codec.base import CodecInfo

__all__ = [
    "CADRE_CODE_COMMIT",
    "CADRE_SOURCE_ENV",
    "CADRE_SOURCE_SHA256",
    "CadreCodec",
    "CadreSourceError",
    "import_cadre",
    "load",
]

CADRE_SOURCE_ENV: Final = "DRIVEDIT_CADRE_SRC"
CADRE_CODE_COMMIT: Final = "3b07314239bf73fe952a98aab78ae03e1937d8bf"
# sha256 of the files of kyutai-labs/cadre-autoencoders-and-world-models at CADRE_CODE_COMMIT that loading imports.
CADRE_SOURCE_SHA256: Final = {
    "model_autoencoder.py": "e82381ff6c647277b09ebae8a2878baad07e44136538d09212dd9a0a52be28eb",
    "util/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "util/attention.py": "f1dd5d154d0f94acb144862fd5eb4d09b4430f02733628675a9192b235442af1",
    "util/hub.py": "5c0d520c650ec939691a69589284900d92bc7f95f5e3db6ffaa31c202222da09",
}
# The CADRE checkpoint's decoder trains on 16-frame clips (config num_frames); decode_clip takes at most that many.
MAX_CLIP_FRAMES: Final = 16


class CadreSourceError(RuntimeError):
    """The CADRE source checkout is missing, differs from the pinned commit, or clashes with loaded modules."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exec_module(name: str, path: Path, *, package_dir: Path | None = None) -> ModuleType:
    locations = [str(package_dir)] if package_dir is not None else None
    spec = importlib.util.spec_from_file_location(name, path, submodule_search_locations=locations)
    if spec is None or spec.loader is None:
        raise CadreSourceError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def import_cadre(source_dir: Path) -> ModuleType:
    """CADRE's ``model_autoencoder`` module from the checkout ``source_dir``, after checking every file's sha256.

    CADRE's modules import each other by their top-level names (``from util import hub``), so they are registered in
    ``sys.modules`` as ``model_autoencoder`` and ``util``. A module of either name already loaded from elsewhere raises
    instead of being replaced.
    """
    root = source_dir.resolve()
    for relative, digest in CADRE_SOURCE_SHA256.items():
        path = root / relative
        if not path.is_file():
            raise CadreSourceError(
                f"{path} is missing: {CADRE_SOURCE_ENV} must name a CADRE checkout at {CADRE_CODE_COMMIT}"
            )
        if _sha256(path) != digest:
            raise CadreSourceError(f"{path} differs from CADRE commit {CADRE_CODE_COMMIT} (sha256 mismatch)")
    for name, expected in (
        ("util", root / "util" / "__init__.py"),
        ("model_autoencoder", root / "model_autoencoder.py"),
    ):
        loaded = sys.modules.get(name)
        if loaded is not None and Path(getattr(loaded, "__file__", "") or "").resolve() != expected:
            raise CadreSourceError(f"a different module named {name!r} is already imported ({loaded!r})")
    if "util" not in sys.modules:
        _exec_module("util", root / "util" / "__init__.py", package_dir=root / "util")
    if "model_autoencoder" in sys.modules:
        return sys.modules["model_autoencoder"]
    return _exec_module("model_autoencoder", root / "model_autoencoder.py")


def _source_dir() -> Path:
    value = os.environ.get(CADRE_SOURCE_ENV)
    if not value:
        raise CadreSourceError(
            f"{CADRE_SOURCE_ENV} is not set; it must name a CADRE checkout at {CADRE_CODE_COMMIT} "
            "(see the drivedit.codec.cadre docstring)"
        )
    return Path(value)


class CadreCodec(FrozenCodec):
    """A tau=1 CADRE autoencoder at its training resolution, 288x512 (see the module docstring).

    ``model`` is a ``model_autoencoder.CADRE`` already on ``device`` and in ``dtype``, in eval mode.
    """

    def __init__(
        self,
        info: CodecInfo,
        model: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        decode_steps: int = 1,
        decode_seed: int = 0,
    ) -> None:
        super().__init__(info, device=device, dtype=dtype)
        config = model.config
        if config.tau != 1:
            raise ValueError(f"{info.codec_id}: only tau=1 cells encode one latent per frame, got tau={config.tau}")
        if decode_steps < 1:
            raise ValueError(f"decode_steps must be >= 1, got {decode_steps}")
        self._model = model
        self._resolution: tuple[int, int] = tuple(config.resolution)
        grid_h, grid_w = config.latent_grid
        self._latent: tuple[int, int, int] = (int(config.latent_dim), int(grid_h), int(grid_w))
        self._mean = float(model.latent_mean)
        self._std = float(model.latent_std)
        self.decode_steps = decode_steps
        self.decode_seed = decode_seed

    def latent_shape(self, height: int, width: int) -> tuple[int, int, int]:
        if (height, width) != self._resolution:
            raise ValueError(
                f"{self.codec_id} takes {self._resolution[0]}x{self._resolution[1]} frames only, got {height}x{width}"
            )
        return self._latent

    def image_size(self, height: int, width: int) -> tuple[int, int]:
        if (height, width) != self._latent[1:]:
            raise ValueError(f"{self.codec_id} latents are {self._latent[1]}x{self._latent[2]}, got {height}x{width}")
        return self._resolution

    def _encode_unit(self, x: torch.Tensor) -> torch.Tensor:
        # x is CADRE.preprocess's output for a frame already at 288x512: u8 / 255 mapped to [-1, 1].
        z: torch.Tensor = self._model.encode(x[:, None])  # [N, T=1, C, h, w]
        return (z[:, 0] - self._mean) / self._std

    def _sample(self, z: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, h, w]`` standardised latents -> ``[B, T, 3, H, W]`` in [-1, 1]."""
        generator = torch.Generator(device=self.device).manual_seed(self.decode_seed)
        decoded: torch.Tensor = self._model.decode(z * self._std + self._mean, self.decode_steps, generator=generator)
        return decoded

    def _decode_unit(self, z: torch.Tensor) -> torch.Tensor:
        return self._sample(z[:, None])[:, 0]

    def decode_clip(self, latents: torch.Tensor) -> torch.Tensor:
        """``[T, C, h, w]`` consecutive latents -> ``[T, 3, H, W]`` uint8, decoded as one causal clip (T <= 16)."""
        if not latents.dtype.is_floating_point or latents.ndim != 4 or tuple(latents.shape[1:]) != self._latent:
            raise ValueError(
                f"{self.codec_id}: latents must be floating [T, *{self._latent}], got "
                f"{latents.dtype} {tuple(latents.shape)}"
            )
        if not 1 <= latents.shape[0] <= MAX_CLIP_FRAMES:
            raise ValueError(f"decode_clip takes 1-{MAX_CLIP_FRAMES} frames, got {latents.shape[0]}")
        self._check_device(latents)
        with torch.no_grad():
            x = self._sample(latents.to(self.dtype)[None])[0]
        return unit_to_images(x)


def load(codec_id: str, *, device: torch.device, dtype: torch.dtype) -> Codec:
    """Build ``codec_id`` (``cadre-1x16x16`` or ``cadre-1x48x48``) from the pinned checkpoint and CADRE checkout."""
    info = CODECS[codec_id]
    if info.module != "cadre":
        raise KeyError(f"{codec_id} is not a CADRE codec")
    module = import_cadre(_source_dir())
    snapshot = fetch_weights(info.weights)
    with torch.device(device):
        model = module.CADRE.from_pretrained(snapshot)
    model = model.to(device=device, dtype=dtype).eval()
    return CadreCodec(info, model, device=device, dtype=dtype)
