"""The Codec interface, its pixel conventions, and the registry of frozen pretrained tokenizers (DD-19).

A codec maps features_v1 ``image`` frames to latent grids and back. Every arm follows the same conventions:

- ``encode(images)``: ``images`` is ``[N, 3, H, W]`` uint8 RGB (the channels ``image_r``, ``image_g``, ``image_b`` of
  features_v1 ``image``, which is ``[T, 3, 288, 512]``), on ``codec.device``. It returns ``[N, C, h, w]`` float32 on
  ``codec.device``, in the codec's own published standardised latent space (each arm's docstring names it). Frames
  are encoded independently, one latent per frame: at 5 Hz there is no temporal compression (MODELING_PLAN §3,
  delta-2026-11 §2).
- ``decode(latents)``: ``[N, C, h, w]`` floating point on ``codec.device``, cast to ``codec.dtype``, to
  ``[N, 3, H, W]`` uint8 RGB on ``codec.device``.
- Pixels map to the [-1, 1] range all three arms train on as ``x = u8 / 127.5 - 1``, and back as
  ``round((x + 1) * 127.5)`` clamped to [0, 255]; the round trip is exact for every uint8 value.
- Nothing moves implicitly: an input on another device, of another dtype or of an unsupported shape raises.
  ``device`` and ``dtype`` (the compute dtype of the frozen weights and activations) are fixed at construction.
- encode and decode run under ``torch.no_grad()``; the weights never train.

``get_codec(codec_id, device=..., dtype=...)`` builds an arm by id. ``CODECS`` carries each arm's pinned weight source
and licence without loading anything, so tools can report provenance and check the local cache first.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast, runtime_checkable

import huggingface_hub
import torch
from huggingface_hub.errors import LocalEntryNotFoundError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CODECS",
    "Codec",
    "CodecInfo",
    "FrozenCodec",
    "WeightSource",
    "codec_ids",
    "fetch_weights",
    "get_codec",
    "images_to_unit",
    "resolve_device",
    "unit_to_images",
    "weights_cached",
]


@dataclass(frozen=True)
class WeightSource:
    """Where a codec's frozen weights come from, pinned to one Hub commit.

    ``files`` are the repository files the codec reads (the ``allow_patterns`` of its snapshot download), relative to
    the repository root. ``code`` names the network code that interprets them. ``licence`` is the weights' licence as
    the model card states it; arms with extra terms spell them out in their module docstring.
    """

    repo_id: str
    revision: str
    files: tuple[str, ...]
    licence: str
    code: str


@dataclass(frozen=True)
class CodecInfo:
    """A registered codec: its id, the ``drivedit.codec`` module that implements it, and its weight source."""

    codec_id: str
    module: str
    weights: WeightSource


@runtime_checkable
class Codec(Protocol):
    """A frozen image tokenizer. Shapes and conventions are in the module docstring."""

    @property
    def codec_id(self) -> str: ...

    @property
    def weights(self) -> WeightSource: ...

    @property
    def device(self) -> torch.device: ...

    @property
    def dtype(self) -> torch.dtype: ...

    def latent_shape(self, height: int, width: int) -> tuple[int, int, int]:
        """``(C, h, w)`` of the latent of one ``height x width`` frame; ValueError for a size the codec cannot take."""
        ...

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """``[N, 3, H, W]`` uint8 RGB -> ``[N, C, h, w]`` float32."""
        ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``[N, C, h, w]`` floating point -> ``[N, 3, H, W]`` uint8 RGB."""
        ...


def resolve_device(device: torch.device | str) -> torch.device:
    """``device`` with an explicit index for CUDA (``cuda`` -> ``cuda:<current>``), so equality checks are exact."""
    resolved = torch.device(device)
    index: int | None = resolved.index  # None at runtime for a bare "cuda"; the torch stubs say int
    if resolved.type == "cuda" and index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return resolved


def images_to_unit(images: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """``[N, 3, H, W]`` uint8 -> the same shape in [-1, 1] as ``dtype`` (computed in float32, then cast)."""
    return (images.to(torch.float32) / 127.5 - 1.0).to(dtype)


def unit_to_images(x: torch.Tensor) -> torch.Tensor:
    """``[N, 3, H, W]`` in [-1, 1] (any float dtype) -> uint8: ``round((x + 1) * 127.5)`` clamped to [0, 255]."""
    return ((x.to(torch.float32) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)


class FrozenCodec(ABC):
    """The shared half of every arm: input checks, the uint8 conversions, no-grad and dtype handling.

    An arm implements ``latent_shape``, its inverse ``image_size`` and the two transforms on [-1, 1] pixels:
    ``_encode_unit`` (``[N, 3, H, W]`` in ``dtype`` -> ``[N, C, h, w]``) and ``_decode_unit`` (the reverse).
    """

    def __init__(self, info: CodecInfo, *, device: torch.device | str, dtype: torch.dtype) -> None:
        if not dtype.is_floating_point:
            raise ValueError(f"compute dtype must be floating point, got {dtype}")
        self._info = info
        self._device = resolve_device(device)
        self._dtype = dtype

    @property
    def codec_id(self) -> str:
        return self._info.codec_id

    @property
    def weights(self) -> WeightSource:
        return self._info.weights

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @abstractmethod
    def latent_shape(self, height: int, width: int) -> tuple[int, int, int]: ...

    @abstractmethod
    def image_size(self, height: int, width: int) -> tuple[int, int]:
        """``(H, W)`` of the frame an ``h x w`` latent decodes to; ValueError for a grid the codec cannot emit."""

    @abstractmethod
    def _encode_unit(self, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _decode_unit(self, z: torch.Tensor) -> torch.Tensor: ...

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """``[N, 3, H, W]`` uint8 RGB on ``device`` -> ``[N, C, h, w]`` float32 on ``device``."""
        if images.dtype != torch.uint8:
            raise TypeError(f"{self.codec_id}: images must be uint8, got {images.dtype}")
        if images.ndim != 4 or images.shape[1] != 3 or images.shape[0] == 0:
            raise ValueError(f"{self.codec_id}: images must be [N>0, 3, H, W], got {tuple(images.shape)}")
        self._check_device(images)
        expected = self.latent_shape(images.shape[2], images.shape[3])
        with torch.no_grad():
            z = self._encode_unit(images_to_unit(images, self._dtype))
        if tuple(z.shape) != (images.shape[0], *expected):
            raise RuntimeError(f"{self.codec_id}: encoder returned {tuple(z.shape)}, expected [N, *{expected}]")
        return z.to(torch.float32)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``[N, C, h, w]`` floating point on ``device`` -> ``[N, 3, H, W]`` uint8 RGB on ``device``."""
        if not latents.dtype.is_floating_point:
            raise TypeError(f"{self.codec_id}: latents must be floating point, got {latents.dtype}")
        if latents.ndim != 4 or latents.shape[0] == 0:
            raise ValueError(f"{self.codec_id}: latents must be [N>0, C, h, w], got {tuple(latents.shape)}")
        self._check_device(latents)
        height, width = self.image_size(latents.shape[2], latents.shape[3])
        expected = self.latent_shape(height, width)
        if tuple(latents.shape[1:]) != expected:
            raise ValueError(f"{self.codec_id}: latents must be [N, *{expected}], got {tuple(latents.shape)}")
        with torch.no_grad():
            x = self._decode_unit(latents.to(self._dtype))
        if tuple(x.shape) != (latents.shape[0], 3, height, width):
            raise RuntimeError(
                f"{self.codec_id}: decoder returned {tuple(x.shape)}, expected [N, 3, {height}, {width}]"
            )
        return unit_to_images(x)

    def _check_device(self, tensor: torch.Tensor) -> None:
        if tensor.device != self._device:
            raise ValueError(
                f"{self.codec_id}: input is on {tensor.device}, the codec on {self._device}; move it first"
            )


class CodecLoader(Protocol):
    """The ``load`` function every arm module defines."""

    def __call__(self, codec_id: str, *, device: torch.device, dtype: torch.dtype) -> Codec: ...


def _registry(*infos: CodecInfo) -> Mapping[str, CodecInfo]:
    return MappingProxyType({info.codec_id: info for info in infos})


# The bake-off arms of MODELING_PLAN §3 / delta-2026-11 §2, pinned to the Hub commits verified on 2026-11-05.
CODECS: Final[Mapping[str, CodecInfo]] = _registry(
    CodecInfo(
        "cadre-1x16x16",
        "cadre",
        WeightSource(
            repo_id="kyutai/cadre-1x16x16",
            revision="bfd521d77858b5e8d663dbdffb8fa315fe4bc6bf",
            files=("config.json", "model.safetensors", "LICENSE-DINOv3"),
            licence="CC-BY-NC-4.0 + DINOv3 License (frozen DINOv3 ViT-B/16 inside)",
            code="drivedit.codec._vendor.cadre (MIT, CADRE @3b07314) + facebookresearch/dinov3@1e358a23 (local)",
        ),
    ),
    CodecInfo(
        "cadre-1x48x48",
        "cadre",
        WeightSource(
            repo_id="kyutai/cadre-1x48x48",
            revision="5e364c9706aa3de4b0453a4ea0ea544e15bcb6e0",
            files=("config.json", "model.safetensors", "LICENSE-DINOv3"),
            licence="CC-BY-NC-4.0 + DINOv3 License (frozen DINOv3 ViT-B/16 inside)",
            code="drivedit.codec._vendor.cadre (MIT, CADRE @3b07314) + facebookresearch/dinov3@1e358a23 (local)",
        ),
    ),
    CodecInfo(
        "flux2-vae",
        "flux2",
        WeightSource(
            repo_id="black-forest-labs/FLUX.2-klein-4B",
            revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
            files=("vae/config.json", "vae/diffusion_pytorch_model.safetensors"),
            licence="Apache-2.0",
            code="diffusers AutoencoderKLFlux2",
        ),
    ),
    CodecInfo(
        "wan21-vae",
        "wan21",
        WeightSource(
            repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            revision="0fad780a534b6463e45facd96134c9f345acfa5b",
            files=("vae/config.json", "vae/diffusion_pytorch_model.safetensors"),
            licence="Apache-2.0",
            code="diffusers AutoencoderKLWan",
        ),
    ),
)


def codec_ids() -> tuple[str, ...]:
    """The registered codec ids, in registry order."""
    return tuple(CODECS)


def fetch_weights(source: WeightSource, *, local_only: bool = False) -> Path:
    """The local snapshot directory of ``source`` at its pinned revision, downloaded into ``HF_HOME`` if needed.

    ``local_only=True`` never touches the network: it raises ``FileNotFoundError`` unless every file of ``source``
    is already in the cache.
    """
    try:
        snapshot = Path(
            huggingface_hub.snapshot_download(
                source.repo_id,
                revision=source.revision,
                allow_patterns=list(source.files),
                local_files_only=local_only,
            )
        )
    except LocalEntryNotFoundError as error:
        raise FileNotFoundError(f"{source.repo_id}@{source.revision} is not in the Hugging Face cache") from error
    missing = [name for name in source.files if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source.repo_id}@{source.revision}: {', '.join(missing)} missing in {snapshot}")
    return snapshot


def weights_cached(codec_id: str) -> bool:
    """True when every weight file of ``codec_id`` is in the local Hugging Face cache (no network access)."""
    try:
        fetch_weights(CODECS[codec_id].weights, local_only=True)
    except FileNotFoundError:
        return False
    return True


def get_codec(codec_id: str, *, device: torch.device | str, dtype: torch.dtype) -> Codec:
    """Build the registered codec ``codec_id`` on ``device`` with compute dtype ``dtype``.

    Downloads its pinned weights on first use (into ``HF_HOME``). The arm module is imported here, not at package
    import, because each arm pulls in its own third-party code (diffusers, or the CADRE source checkout).
    """
    info = CODECS.get(codec_id)
    if info is None:
        raise KeyError(f"unknown codec {codec_id!r}; registered: {', '.join(CODECS)}")
    module = importlib.import_module(f"drivedit.codec.{info.module}")
    load = cast("CodecLoader", module.load)
    return load(codec_id, device=resolve_device(device), dtype=dtype)
