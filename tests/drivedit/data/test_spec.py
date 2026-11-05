"""drivedit.data.spec: the features_v1 registry, the access table and the C24 forbidden-input check."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from drivedit.data.spec import (
    ROLES,
    AccessError,
    FeatureSpec,
    SpecError,
    load_spec,
    registry_problems,
)

from .conftest import REPO

FEATURES_JSON = REPO / "docs" / "capture_contract" / "features_v1.json"


def test_spec_matches_the_registry_file(spec: FeatureSpec) -> None:
    raw = json.loads(FEATURES_JSON.read_text("utf-8"))
    assert spec.tensor_names == tuple(t["name"] for t in raw["tensors"])
    for t in raw["tensors"]:
        ts = spec.tensor(t["name"])
        assert list(ts.shape) == t["shape"]
        assert ts.cls == t["class"]
        assert list(ts.channel_names) == [c["name"] for c in t["channels"]]
        for role in ROLES:
            ra = ts.role(role)
            assert (ra.input, ra.max_offset, ra.target) == tuple(
                t["access"][role][k] for k in ("input", "max_offset", "target")
            )
    assert spec.decision_hz == 5
    assert spec.physics_steps_per_decision == 6
    assert registry_problems(spec) == []


def test_world_model_conditions_on_action_cmd_at_offset_0_only(spec: FeatureSpec) -> None:
    access = spec.access("world_model", "action_cmd")
    access.require_input(0)
    access.require_input(-3)
    with pytest.raises(AccessError, match="offset <= 0"):
        access.require_input(1)
    spec.access("world_model", "state").require_input(0)
    with pytest.raises(AccessError):
        spec.access("world_model", "state").require_input(1)  # never realised future state


def test_policy_sees_actions_only_as_history(spec: FeatureSpec) -> None:
    for name in ("action_cmd", "kappa_target", "action_exec", "exec_brake"):
        access = spec.access("policy", name)
        access.require_input(-1)
        with pytest.raises(AccessError):
            access.require_input(0)
    spec.access("policy", "nav").require_input(0)
    spec.access("policy", "action_label").require_target()
    with pytest.raises(AccessError):
        spec.access("policy", "action_label").require_input(-1)


def test_targets_and_masks(spec: FeatureSpec) -> None:
    for name in ("future_pose", "contact_events", "hindsight_anchors", "road_context"):
        spec.access("world_model", name).require_target()
    with pytest.raises(AccessError):
        spec.access("world_model", "action_exec").require_target()  # an input, not a world-model target
    for role in ROLES:
        access = spec.access(role, "masks")
        assert not access.input
        assert not access.target


def test_every_c24_name_is_refused_for_every_role(spec: FeatureSpec) -> None:
    raw = json.loads(FEATURES_JSON.read_text("utf-8"))
    names = raw["c24_forbidden_inputs"]["names"]
    assert names
    for role in ROLES:
        for name in [*names, "hindsight_anchors", "future_pose", "eval_only"]:
            access = spec.access(role, name)
            assert access.c24_forbidden
            assert not access.input
            with pytest.raises(AccessError, match="C24"):
                access.require_input(-1)


def test_c24_patterns_refuse_unregistered_names(spec: FeatureSpec) -> None:
    for name in ("goal_x", "anchor_8s_x", "fut_speed", "speed_world_x"):
        access = spec.access("teacher", name)
        assert access.c24_forbidden
        assert access.tensor is None
        with pytest.raises(AccessError):
            access.require_input(0)


def test_unknown_names_and_roles_fail_closed(spec: FeatureSpec) -> None:
    with pytest.raises(KeyError):
        spec.access("world_model", "steering_wheel")
    with pytest.raises(KeyError):
        spec.access("critic", "state")


def test_registry_problems_catch_a_forbidden_input(spec: FeatureSpec) -> None:
    state = spec.tensor("state")
    leaky = dataclasses.replace(state, channels=(*state.channels, dataclasses.replace(state.channels[0], name="pos_x")))
    absolute = dataclasses.replace(
        state, channels=(dataclasses.replace(state.channels[0], absolute=True), *state.channels[1:])
    )
    for bad in (leaky, absolute):
        broken = dataclasses.replace(spec, tensors=tuple(bad if t.name == "state" else t for t in spec.tensors))
        assert registry_problems(broken)


def test_shapes(spec: FeatureSpec) -> None:
    assert spec.shape("action_exec", 7, decision_hz=5) == (7, 6, 4)
    assert spec.shape("action_exec", 7, decision_hz=10) == (7, 3, 4)  # offsets.override_10hz
    assert spec.shape("future_pose", 7, decision_hz=5, horizon=30) == (7, 30, 3)
    assert spec.steps_per_decision(10) == 3
    with pytest.raises(SpecError):
        spec.shape("future_pose", 7, decision_hz=5)
    with pytest.raises(SpecError):
        spec.steps_per_decision(20)


def test_missing_registry_file_names_the_path(tmp: Path) -> None:
    with pytest.raises(SpecError, match=r"features_v1\.json not found"):
        load_spec(tmp / "features_v1.json")
