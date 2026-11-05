"""Evaluation: action recoverability, response and sign tests, state probes, and the engine-facing evaluators.

The core metrics are numpy-only. Learned probes (for example an inverse-dynamics model) import torch inside their
own modules, never at package import.

- ``e0``: the E0 telemetry check (does ``action_cmd`` improve next-step prediction over ``state``?) and its CLI;
- ``control``: the yaw-gain test through llc_v1, the response slope, opposite-action sign accuracy, the brake test,
  and ``grade`` (a model result beside its real-data ceiling);
- ``probes``: linear state probes and the explained action information;
- ``linear``, ``strata``: closed-form ridge with segment-respecting cross-validation, and the speed bins;
- ``idm`` (torch): the inverse-dynamics model and CADRE-style ARR with a donor-action control.
"""
