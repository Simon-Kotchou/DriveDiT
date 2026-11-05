"""DriveDiT: an action-conditioned world model of Arma Reforger driving, and a policy trained inside it.

Subpackages:

- ``drivedit.data``: the features_v1 dataset over harvested dc-2.0 sessions. numpy only; torch appears only in
  ``drivedit.data.torch``.
- ``drivedit.codec``: frozen pretrained image tokenizers and the one-time latent pre-encoder (torch).
- ``drivedit.model``: the causal latent DiT world model (torch).
- ``drivedit.train``: training loops, noise schedules and run configs (torch).
- ``drivedit.eval``: controllability metrics, state probes and the engine-facing evaluators.

Importing ``drivedit``, ``drivedit.data`` or ``drivedit.eval`` never imports torch. The capture tooling and its CI job
stay torch-free.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("drivedit")
except PackageNotFoundError:  # a source tree without an install
    __version__ = "0+unknown"

__all__ = ["__version__"]
