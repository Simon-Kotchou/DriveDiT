"""features_v1 dataset: harvested dc-2.0 sessions to model-ready windows.

Builds the model-boundary tensors of ``docs/capture_contract/features_v1.json`` from the dc-2.0 reader, writes them as
memory-mapped shards, and samples episode-aware windows. Offsets and the C24 forbidden inputs are enforced here.

numpy only. torch is confined to ``drivedit.data.torch`` (a thin Dataset over the shards), and ruff bans it elsewhere
in this package.
"""
