"""Evaluation: action recoverability, response and sign tests, state probes, and the engine-facing evaluators.

The core metrics are numpy-only. Learned probes (for example an inverse-dynamics model) import torch inside their
own modules, never at package import.
"""
