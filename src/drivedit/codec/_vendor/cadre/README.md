# Vendored CADRE autoencoder code

- **Source:** https://github.com/kyutai-labs/cadre-autoencoders-and-world-models
- **Commit:** `3b07314239bf73fe952a98aab78ae03e1937d8bf` (2026-10-20)
- **Licence:** MIT, see `LICENSE` (copied unchanged from the repository root). The CADRE *weights* are a separate
  matter: CC-BY-NC-4.0, plus the DINOv3 License for the DINOv3 weights inside them. They are not in this directory.

## What was copied

Only what CADRE's autoencoder needs to encode and decode, imported by `drivedit.codec.cadre`:

| File here | Upstream file |
|---|---|
| `model_autoencoder.py` | `model_autoencoder.py` |
| `util/attention.py` | `util/attention.py` |
| `util/hub.py` | `util/hub.py` (unchanged) |

`util/__init__.py` is empty upstream too. The world model, training, evaluation, demo, data and the other `util`
modules were not copied.

## Edits

1. **Relative imports.** `from util import hub` and `from util.attention import ...` became `from .util import ...`,
   so nothing resolves through `sys.path` or a global `util` package.
2. **DINOv3 is passed in.** Upstream builds its frozen DINOv3 ViT-B/16 with
   `torch.hub.load("facebookresearch/dinov3:1e358a23...", trust_repo=True, skip_validation=True)`, which downloads and
   runs unverified code. `DINOV3_HUB` and `load_dinov3()` were removed; `Encoder`, `CADRE`, `CADRE.from_checkpoint` and
   `CADRE.from_pretrained` take the module as a `dino` argument instead. drivedit builds it from a hash-verified local
   checkout of that same commit (`drivedit/codec/dinov3.py`). The `Encoder` docstring says so.
3. **The opt-in tiny temporal attention was removed** from `util/attention.py` (the `tiny_temporal_attention` flag and
   its branch). It was off by default, and its module `util/tiny_temporal_attention.py` is not vendored.

Everything else is byte-for-byte upstream. Compare with
`git diff 3b07314 -- model_autoencoder.py util/attention.py util/hub.py` in a checkout.
