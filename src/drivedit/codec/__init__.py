"""Frozen pretrained image tokenizers behind one interface, and the latent pre-encoder.

Each frame is encoded once, offline, into a latent grid that the world model trains on. The codec stays frozen: we
never train our own VAE. Candidates are chosen by the world model trained on them, not by reconstruction quality.

- ``drivedit.codec.base``: the ``Codec`` Protocol, its uint8 conventions, and the registry (``get_codec``, ``CODECS``).
- ``drivedit.codec.cadre``, ``.flux2``, ``.wan21``: the bake-off arms, imported by ``get_codec`` on demand.
- ``drivedit.codec.encode``: ``encode_session`` writes a shard's latents (DRIVEDIT_INTERFACES §3), and its CLI.

Needs the ``model`` and ``codec`` extras. Importing this package imports torch, numpy and OpenCV, but neither
huggingface_hub (imported by the first ``fetch_weights``, after an entry point has routed its cache into the write
boundary; ``drivedit.process_paths``) nor the arms' third-party code (diffusers, the CADRE checkout).
"""

from drivedit.codec.base import CODECS, Codec, CodecInfo, WeightSource, codec_ids, get_codec, weights_cached
from drivedit.codec.encode import encode_session

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
