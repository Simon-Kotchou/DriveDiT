"""features_v1 dataset: harvested dc-2.0 sessions to model-ready windows.

Builds the model-boundary tensors of ``docs/capture_contract/features_v1.json`` from the dc-2.0 reader, writes them as
memory-mapped shards, and samples episode-aware windows. Offsets and the C24 forbidden inputs are enforced here.

- ``spec``: ``load_spec()`` and ``FeatureSpec.access(role, name)``, the features_v1 registry and its C24 check.
- ``build``: ``drivedit.data.build.build_session()`` and ``python -m drivedit.data.build``, one session to one shard.
  It is not re-exported here, so running the module as a script loads it once.
- ``shard``: ``Shard``, ``Latents``, ``LatentStats`` and ``WindowIndex``, read-only access and episode-aware windows.

numpy only. torch is confined to ``drivedit.data.torch`` (a thin Dataset over the shards), and ruff bans it elsewhere
in this package.
"""

from drivedit.data.shard import Latents, LatentStats, Shard, ShardError, Window, WindowIndex
from drivedit.data.spec import Access, AccessError, FeatureSpec, SpecError, TensorSpec, load_spec

__all__ = [
    "Access",
    "AccessError",
    "FeatureSpec",
    "LatentStats",
    "Latents",
    "Shard",
    "ShardError",
    "SpecError",
    "TensorSpec",
    "Window",
    "WindowIndex",
    "load_spec",
]
