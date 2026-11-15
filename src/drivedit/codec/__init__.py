"""Frozen pretrained image tokenizers behind one interface, and the latent pre-encoder.

Each frame is encoded once, offline, into a latent grid that the world model trains on. The codec stays frozen: we
never train our own VAE. Candidates are chosen by the world model trained on them, not by reconstruction quality.

- ``drivedit.codec.base``: the ``Codec`` Protocol, its uint8 conventions, and the registry (``get_codec``, ``CODECS``).
- ``drivedit.codec.cadre``, ``.flux2``, ``.wan21``: the bake-off arms, imported by ``get_codec`` on demand.
- ``drivedit.codec.encode``: ``encode_session`` writes a shard's latents (DRIVEDIT_INTERFACES §3), and its CLI.

Needs the ``model`` and ``codec`` extras. The names below load on first use (PEP 562), not with the package: importing
``drivedit.codec`` imports nothing heavy, so ``python -m drivedit.codec.encode`` can check what importing torch would
write before torch is imported (DD-23, ``drivedit.process_paths``). ``drivedit.codec.base`` imports torch;
huggingface_hub is imported by the first ``fetch_weights``, after an entry point has routed its cache into the write
boundary, and the arms' third-party code (diffusers, the CADRE checkout) by ``get_codec``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from drivedit.codec.base import CODECS, Codec, CodecInfo, WeightSource, codec_ids, get_codec, weights_cached
    from drivedit.codec.encode import encode_session

#: exported name -> the module that defines it.
_EXPORTS = {
    **dict.fromkeys(
        ("CODECS", "Codec", "CodecInfo", "WeightSource", "codec_ids", "get_codec", "weights_cached"),
        "drivedit.codec.base",
    ),
    "encode_session": "drivedit.codec.encode",
}

__all__ = [
    "CODECS",
    "Codec",
    "CodecInfo",
    "WeightSource",
    "codec_ids",
    "encode_session",
    "get_codec",
    "weights_cached",
]


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    """The module's own names and the lazy exports (they are not in globals() until first used)."""
    return sorted({*globals(), *_EXPORTS})
