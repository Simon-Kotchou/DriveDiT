"""Frozen pretrained image tokenizers behind one interface, and the latent pre-encoder.

Each frame is encoded once, offline, into a latent grid that the world model trains on. The codec stays frozen: we
never train our own VAE. Candidates are chosen by the world model trained on them, not by reconstruction quality.
"""
