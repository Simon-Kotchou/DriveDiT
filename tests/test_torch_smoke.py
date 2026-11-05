"""Smoke test of the `model` extra, until the new model package brings its own tests.

The March 2026 model stack was removed from main in DD-16 (tag archive/model-2026-03). This module keeps the CI model
job meaningful in the meantime: `uv sync --extra model --extra cpu` must install, from uv.lock, a torch that imports
and computes, together with the extra's other packages, and the `torch` marker must reach the job
(tests/conftest.py TORCH_TEST_MODULES). Without torch the module is not collected.
"""

import importlib

import torch
import torch.nn.functional as F


def test_model_extra_imports():
    for name in ("einops", "torchmetrics", "tqdm"):
        importlib.import_module(name)
    assert torch.__version__ >= "2.2"


def test_causal_attention_matches_sdpa():
    """Explicit softmax attention under a causal mask equals torch's scaled_dot_product_attention."""
    gen = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8, generator=gen) for _ in range(3))  # [B, H, T, D]
    causal = torch.ones(16, 16, dtype=torch.bool).tril()
    scores = torch.einsum("bhtd,bhsd->bhts", q, k) / q.size(-1) ** 0.5
    expected = torch.einsum("bhts,bhsd->bhtd", scores.masked_fill(~causal, float("-inf")).softmax(-1), v)
    actual = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(actual, expected)
