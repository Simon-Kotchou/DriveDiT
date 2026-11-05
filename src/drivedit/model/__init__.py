"""The causal latent DiT world model.

The model predicts the next frame's latent from past latents, the applied ``action_cmd`` and the vehicle state, one
latent frame per 5 Hz decision step. It is trained with per-frame noise (diffusion forcing) and has auxiliary heads
for privileged targets (future pose, contacts) that are never inputs.
"""
