"""The torch Dataset over features_v1 windows and their pre-encoded latents (the one torch module of drivedit.data).

    index = WindowIndex(shards, length=10)
    data = LatentWindowDataset(index, "cadre-1x16x16", stats=stats)
    item = data[0]
    item["latents"]          # [L, C, h, w] float32, (x - mean) / std per channel; 0 where valid is False
    item["valid"]            # [L] bool: the latent row exists (valid.npy)
    item["action_cmd"]       # [L, 3] float32, item["state"] [L, 4]: aligned at t, unshifted
    item["future_pose"]      # [L, H, 3] float32 aux target (NaN = no target at that step and horizon)
    item["contact_events"]   # [L, 2] float32 aux target
    item["masks"]            # [L, 5] uint8: policy_valid, wm_valid, warmup, env_transition, teleport

Every channel is delivered aligned at its decision step t (DRIVEDIT_INTERFACES section 1): the world model shifts
itself (frame f is conditioned on ``action_cmd[f-1]`` and ``state[f-1]``). So each requested input channel must be
an input of ``role`` at offset 0 (``FeatureSpec.access``): C24 forbidden names, aux targets, eval_only channels and
the policy's history-only actions are refused here. Each aux name must be a target of ``role``. A non-finite value in
an input channel raises, except the empty action_exec entries past P_t+1 (whole entries of NaN, ``EMPTY_ENTRIES``);
aux targets keep NaN, which the loss masks. ``masks`` rides along for the loss (never an
input). A build-scoped channel (``action_exec``) is never pooled across build_keys (features_v1 rules).

The latent normalisation ``stats`` comes from the caller: drivedit.train computes it over the training split only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from drivedit.data.shard import Latents, LatentStats, ShardError, Window, WindowIndex
from drivedit.data.spec import FeatureSpec, load_spec

# Input tensors with entries the contract leaves empty: action_exec holds the steps P_t + 1 .. P_t + N, and the
# entries past P_t+1 are empty (features_v1 action_exec rule; NaN in the shard) whenever n_t < N.
EMPTY_ENTRIES = ("action_exec",)


class LatentWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """One window of ``index`` per item: latents, valid, the requested channels, the aux targets and masks."""

    def __init__(
        self,
        index: WindowIndex,
        codec_id: str,
        *,
        stats: LatentStats,
        channels: Sequence[str] = ("action_cmd", "state"),
        aux: Sequence[str] = ("future_pose", "contact_events"),
        role: str = "world_model",
        spec: FeatureSpec | None = None,
    ) -> None:
        spec = spec if spec is not None else load_spec()
        for name in channels:
            spec.access(role, name).require_input(0)
        for name in aux:
            spec.access(role, name).require_target()
        reserved = {"latents", "valid", "masks"}
        names = [*channels, *aux]
        if len(set(names)) != len(names) or reserved & set(names):
            raise ValueError(f"channel and aux names must be distinct and not {sorted(reserved)}: {names}")
        scoped = [n for n in channels if spec.tensor(n).build_scoped]
        keys = {s.build_key for s in index.shards}
        if scoped and (len(keys) > 1 or None in keys):
            raise ValueError(f"{scoped} are build-scoped and never pooled across build_keys; the index has {keys}")
        for shard in index.shards:
            for name in [*names, "masks"]:
                if not shard.has(name):
                    raise ShardError(f"{shard.path} has no {name} ({shard.meta.get('missing', {}).get(name)})")
        self.index = index
        self.codec_id = codec_id
        self.stats = stats
        self.channels = tuple(channels)
        self.aux = tuple(aux)
        self.role = role
        self._latents: dict[str, Latents] = {s.session_id: s.latents(codec_id) for s in index.shards}
        shapes = {lat.latent_shape for lat in self._latents.values()}
        if len(shapes) > 1:
            raise ShardError(f"{codec_id}: latent shapes differ across sessions: {sorted(shapes)}")
        if shapes and next(iter(shapes))[0] != len(stats.mean):
            raise ValueError(f"{codec_id} latents have {next(iter(shapes))[0]} channels, the stats {len(stats.mean)}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        window = self.index[i]
        latents = self._latents[window.shard.session_id]
        valid = np.array(latents.valid[window.start : window.stop], dtype=np.bool_)
        x = self.stats.normalise(np.asarray(latents.latents[window.start : window.stop]))
        x[~valid] = 0.0
        item = {"latents": torch.from_numpy(x), "valid": torch.from_numpy(valid)}
        for name in self.channels:
            values = window.take(name).astype(np.float32)
            finite = np.isfinite(values)
            if name in EMPTY_ENTRIES:
                # an entry is empty (all NaN) or whole; the empty ones pass through for the model to mask
                entries = finite.reshape(*finite.shape[:2], -1)
                ok = bool((entries.all(axis=-1) | ~entries.any(axis=-1)).all())
            else:
                ok = bool(finite.all())
            if not ok:
                raise ValueError(f"{_where(window)}: {name} is not finite inside the window")
            item[name] = torch.from_numpy(values)
        for name in self.aux:
            item[name] = torch.from_numpy(window.take(name).astype(np.float32))
        item["masks"] = torch.from_numpy(window.take("masks"))
        return item


def _where(window: Window) -> str:
    return f"{window.shard.session_id} steps [{window.start}, {window.stop})"
