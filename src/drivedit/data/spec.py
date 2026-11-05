"""The features_v1 registry: tensors, channels, per-role access and the C24 forbidden inputs.

``load_spec()`` parses ``docs/capture_contract/features_v1.json``, the frozen model-boundary registry of contract
dc-2.0 (section 12). Until the package move (DD-17) the file is found through the repository checkout this package
was installed from (``src/drivedit/data/spec.py`` -> ``<repo>/docs/capture_contract/``); a missing file raises
``SpecError`` naming the path.

    spec = load_spec()
    spec.tensor("action_cmd").channel_names       # ("kappa_target", "a_target", "dir_cmd")
    spec.access("world_model", "action_cmd")      # Access(input=True, max_offset=0, target=False, ...)
    spec.access("policy", "action_cmd").require_input(0)   # AccessError: the policy sees it only as history
    spec.access("world_model", "pos_x").require_input(0)   # AccessError: C24 forbidden input

Access is looked up by tensor name or channel name. A name on C24's list (``c24_forbidden_inputs``: its names and
its fnmatch patterns, e.g. ``anchor_*`` or ``goal_*``) is never an input for any role, whatever tensor it belongs
to, and a name the registry does not know raises ``KeyError`` (fail closed: nothing unregistered reaches a model).

Loading re-checks the registry rules that ``access()`` relies on (dc-2.0 section 12; ``tests/test_contract_dc2.py``
checks the same file in CI): no input-class channel is absolute or on C24's list, aux_target and eval_only tensors are
never inputs, eval_only tensors are never targets, and only the teacher has unbounded input offsets.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPEC_SCHEMA = "drivedit.features/1"
FEATURES_VERSION = "features_v1"
CONTRACT_VERSION = "dc-2.0"
CLASSES = ("student_input", "teacher_input", "aux_target", "eval_only")
ROLES = ("world_model", "policy", "teacher")
DEPLOYED_ROLES = ("world_model", "policy")
INPUT_CLASSES = ("student_input", "teacher_input")
SPEC_RELPATH = Path("docs") / "capture_contract" / "features_v1.json"


class SpecError(ValueError):
    """features_v1.json is missing, unreadable, or breaks a registry rule."""


class AccessError(PermissionError):
    """A name is refused as a model input (role, offset or C24) or as a training target."""


def repo_root() -> Path:
    """The repository checkout this package runs from (``src/drivedit/data`` -> three levels up)."""
    return Path(__file__).resolve().parents[3]


def default_spec_path() -> Path:
    """``<repo>/docs/capture_contract/features_v1.json``."""
    return repo_root() / SPEC_RELPATH


@dataclass(frozen=True)
class ChannelSpec:
    """One channel of a tensor, verbatim from features_v1 (``sign`` is ``"right+"`` or None)."""

    name: str
    unit: str
    source: str
    absolute: bool
    build_scoped: bool
    sign: str | None
    desc: str


@dataclass(frozen=True)
class RoleAccess:
    """features_v1 ``access[role]``: input with a maximum offset in decision steps (None = unbounded, teacher only),
    and whether the tensor is a target of that role."""

    input: bool
    max_offset: int | None
    target: bool


@dataclass(frozen=True)
class TensorSpec:
    """One features_v1 tensor. ``shape`` keeps the registry's symbols: ``"T"`` (decision steps), ``"H"`` (the
    future_pose horizon, set by the builder) and ``"-"`` (the eval_only width)."""

    name: str
    shape: tuple[int | str, ...]
    cls: str
    kind: str
    optional: bool
    rule: str
    channels: tuple[ChannelSpec, ...]
    roles: tuple[tuple[str, RoleAccess], ...]

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.channels)

    @property
    def build_scoped(self) -> bool:
        """True when any channel is a raw engine input that is never pooled across build_keys."""
        return any(c.build_scoped for c in self.channels)

    def role(self, role: str) -> RoleAccess:
        for name, access in self.roles:
            if name == role:
                return access
        raise KeyError(f"tensor {self.name} has no access entry for role {role!r}")


@dataclass(frozen=True)
class Access:
    """The answer of ``FeatureSpec.access(role, name)``.

    ``tensor`` is the owning tensor (None for a C24 name or pattern match that no tensor carries, e.g. ``goal_x``).
    ``input`` is False for every C24 forbidden name, whatever the registry says. ``max_offset`` bounds the offsets
    (decision steps, relative to t) at which an input may be read; None with ``input`` means unbounded.
    """

    role: str
    name: str
    tensor: str | None
    cls: str | None
    input: bool
    max_offset: int | None
    target: bool
    c24_forbidden: bool

    def allows_input(self, offset: int) -> bool:
        return self.input and (self.max_offset is None or offset <= self.max_offset)

    def require_input(self, offset: int) -> None:
        """Raise AccessError unless ``name`` may be a ``role`` input at ``offset`` decision steps."""
        if self.c24_forbidden:
            raise AccessError(
                f"{self.name} is a C24 forbidden input (features_v1 c24_forbidden_inputs): never a model input"
            )
        if not self.input:
            raise AccessError(f"{self.name} ({self.cls}) is not a {self.role} input (features_v1 access)")
        if self.max_offset is not None and offset > self.max_offset:
            raise AccessError(
                f"{self.name} is a {self.role} input only at offset <= {self.max_offset}, not {offset} "
                "(features_v1 access)"
            )

    def require_target(self) -> None:
        """Raise AccessError unless ``name`` is a training target of ``role``."""
        if not self.target:
            raise AccessError(f"{self.name} ({self.cls}) is not a {self.role} target (features_v1 access)")


@dataclass(frozen=True)
class Override:
    """features_v1 ``offsets.override_10hz``: the non-default decision rate and the shapes it changes."""

    decision_hz: float
    physics_steps_per_decision: int
    shapes: tuple[tuple[str, tuple[int | str, ...]], ...]


@dataclass(frozen=True)
class FeatureSpec:
    """The parsed features_v1 registry. Look tensors up with ``tensor(name)`` and access with ``access(role, name)``."""

    path: Path
    sha256: str
    features_version: str
    contract_version: str
    decision_hz: float
    physics_steps_per_decision: int
    step_dt_s: float
    override: Override
    rules: tuple[str, ...]
    c24_names: frozenset[str]
    c24_patterns: tuple[str, ...]
    tensors: tuple[TensorSpec, ...]

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tensors)

    def tensor(self, name: str) -> TensorSpec:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(f"features_v1 has no tensor {name!r}")

    def owner(self, channel: str) -> TensorSpec:
        """The tensor that holds a channel name."""
        for t in self.tensors:
            if channel in t.channel_names:
                return t
        raise KeyError(f"features_v1 has no channel {channel!r}")

    def is_c24_forbidden(self, name: str) -> bool:
        """True for a C24 name or a name matching a C24 pattern (fnmatch, case-sensitive)."""
        return name in self.c24_names or any(fnmatch.fnmatchcase(name, p) for p in self.c24_patterns)

    def access(self, role: str, name: str) -> Access:
        """Access of a tensor or channel name for a role. A tensor counts as C24 forbidden when its name or any of its
        channels is. KeyError for an unknown role or an unregistered name that is not a C24 match."""
        if role not in ROLES:
            raise KeyError(f"unknown role {role!r} (features_v1 roles: {', '.join(ROLES)})")
        tensor: TensorSpec | None
        if name in self.tensor_names:
            tensor = self.tensor(name)
            forbidden = self.is_c24_forbidden(name) or any(self.is_c24_forbidden(c) for c in tensor.channel_names)
        else:
            try:
                tensor = self.owner(name)
            except KeyError:
                if not self.is_c24_forbidden(name):
                    raise KeyError(f"{name!r} is neither a features_v1 tensor nor a channel") from None
                return Access(role, name, None, None, False, None, False, True)
            forbidden = self.is_c24_forbidden(name)
        ra = tensor.role(role)
        if forbidden:
            return Access(role, name, tensor.name, tensor.cls, False, None, ra.target, True)
        return Access(role, name, tensor.name, tensor.cls, ra.input, ra.max_offset, ra.target, False)

    def steps_per_decision(self, decision_hz: float) -> int:
        """physics_steps_per_decision N at a decision rate: 6 at the 5 Hz default, 3 at the 10 Hz override."""
        if decision_hz == self.decision_hz:
            return self.physics_steps_per_decision
        if decision_hz == self.override.decision_hz:
            return self.override.physics_steps_per_decision
        raise SpecError(
            f"decision rate {decision_hz} Hz is neither the features_v1 default {self.decision_hz} Hz nor the "
            f"override {self.override.decision_hz} Hz"
        )

    def shape(self, name: str, n_steps: int, *, decision_hz: float, horizon: int | None = None) -> tuple[int, ...]:
        """Concrete shape of a tensor: T = n_steps, H = horizon (future_pose), and the 10 Hz override's shapes."""
        symbolic = self.tensor(name).shape
        if decision_hz != self.decision_hz:
            self.steps_per_decision(decision_hz)
            symbolic = dict(self.override.shapes).get(name, symbolic)
        out: list[int] = []
        for dim in symbolic:
            if isinstance(dim, int):
                out.append(dim)
            elif dim == "T":
                out.append(n_steps)
            elif dim == "H":
                if horizon is None or horizon < 1:
                    raise SpecError(f"{name} has the builder-set axis H; give a horizon >= 1")
                out.append(horizon)
            else:
                raise SpecError(f"{name} has the open axis {dim!r}; it has no fixed shape")
        return tuple(out)


def load_spec(path: str | Path | None = None) -> FeatureSpec:
    """Parse and check features_v1.json (default: the repository's ``docs/capture_contract/features_v1.json``)."""
    p = Path(path) if path is not None else default_spec_path()
    if not p.is_file():
        raise SpecError(
            f"features_v1.json not found at {p}: drivedit.data reads the contract from the repository checkout "
            "(docs/capture_contract/) until the package move (DD-17); run from a checkout or pass its path"
        )
    raw_bytes = p.read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SpecError(f"{p}: not JSON ({e})") from e
    spec = _parse(_obj(raw, "features_v1.json"), p, hashlib.sha256(raw_bytes).hexdigest())
    problems = registry_problems(spec)
    if problems:
        raise SpecError(f"{p}: {len(problems)} registry problem(s), first: {problems[0]}")
    return spec


def registry_problems(spec: FeatureSpec) -> list[str]:
    """The features_v1 rules ``access()`` relies on (dc-2.0 section 12), as messages; empty when they hold."""
    problems: list[str] = []
    for t in spec.tensors:
        if t.cls not in CLASSES:
            problems.append(f"tensor {t.name}: class {t.cls!r}")
        problems += _access_problems(t)
        if t.cls in INPUT_CLASSES:
            for c in t.channels:
                if c.absolute and t.cls == "student_input":
                    problems.append(f"{t.name}.{c.name}: a student_input channel is absolute")
                if spec.is_c24_forbidden(c.name):
                    problems.append(f"{t.name}.{c.name}: an input-class channel is a C24 forbidden input")
    names = [c for t in spec.tensors for c in t.channel_names]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        problems.append(f"channel names used twice: {', '.join(duplicated)}")
    return problems


def _access_problems(t: TensorSpec) -> list[str]:
    problems: list[str] = []
    for role in ROLES:
        try:
            ra = t.role(role)
        except KeyError:
            problems.append(f"tensor {t.name}: no access for role {role}")
            continue
        if ra.input and t.cls in ("aux_target", "eval_only"):
            problems.append(f"tensor {t.name}: {t.cls} is an input of {role}")
        if ra.target and t.cls == "eval_only":
            problems.append(f"tensor {t.name}: eval_only is a target of {role}")
        if ra.input and ra.max_offset is None and role in DEPLOYED_ROLES:
            problems.append(f"tensor {t.name}: unbounded input offsets for the deployed role {role}")
        if ra.input and t.cls == "teacher_input" and role in DEPLOYED_ROLES:
            problems.append(f"tensor {t.name}: teacher_input reaches the deployed role {role}")
    return problems


# ------------------------------------------------------------------------------------------------
# JSON parsing (typed)
# ------------------------------------------------------------------------------------------------
def _obj(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SpecError(f"{where}: not a JSON object")
    return value


def _list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise SpecError(f"{where}: not a JSON list")
    return value


def _str(obj: dict[str, Any], key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str):
        raise SpecError(f"{where}.{key}: not a string")
    return value


def _bool(obj: dict[str, Any], key: str, where: str, default: bool | None = None) -> bool:
    value = obj.get(key, default)
    if not isinstance(value, bool):
        raise SpecError(f"{where}.{key}: not true or false")
    return value


def _number(obj: dict[str, Any], key: str, where: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SpecError(f"{where}.{key}: not a number")
    return float(value)


def _int(obj: dict[str, Any], key: str, where: str) -> int:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{where}.{key}: not an integer")
    return value


def _shape(value: Any, where: str) -> tuple[int | str, ...]:
    dims: list[int | str] = []
    for d in _list(value, where):
        if isinstance(d, bool) or not isinstance(d, int | str):
            raise SpecError(f"{where}: axis {d!r} is neither an integer nor a symbol")
        dims.append(d)
    if not dims or dims[0] != "T":
        raise SpecError(f"{where}: a features_v1 shape starts with T")
    return tuple(dims)


def _role_access(value: Any, where: str) -> RoleAccess:
    a = _obj(value, where)
    max_offset = a.get("max_offset")
    if max_offset is not None and (isinstance(max_offset, bool) or not isinstance(max_offset, int)):
        raise SpecError(f"{where}.max_offset: neither null nor an integer")
    return RoleAccess(_bool(a, "input", where), max_offset, _bool(a, "target", where))


def _channel(value: Any, where: str) -> ChannelSpec:
    c = _obj(value, where)
    sign = c.get("sign")
    if sign is not None and not isinstance(sign, str):
        raise SpecError(f"{where}.sign: not a string")
    return ChannelSpec(
        name=_str(c, "name", where),
        unit=_str(c, "unit", where),
        source=_str(c, "source", where),
        absolute=_bool(c, "absolute", where),
        build_scoped=_bool(c, "build_scoped", where),
        sign=sign,
        desc=_str(c, "desc", where),
    )


def _tensor(value: Any, where: str) -> TensorSpec:
    t = _obj(value, where)
    name = _str(t, "name", where)
    where = f"tensor {name}"
    access = _obj(t.get("access"), f"{where}.access")
    return TensorSpec(
        name=name,
        shape=_shape(t.get("shape"), f"{where}.shape"),
        cls=_str(t, "class", where),
        kind=_str(t, "kind", where),
        optional=_bool(t, "optional", where, default=False),
        rule=_str(t, "rule", where),
        channels=tuple(_channel(c, f"{where}.channels") for c in _list(t.get("channels"), f"{where}.channels")),
        roles=tuple((role, _role_access(a, f"{where}.access.{role}")) for role, a in access.items()),
    )


def _parse(raw: dict[str, Any], path: Path, sha256: str) -> FeatureSpec:
    where = path.name
    if raw.get("schema") != SPEC_SCHEMA:
        raise SpecError(f"{where}: schema {raw.get('schema')!r} is not {SPEC_SCHEMA}")
    if raw.get("features_version") != FEATURES_VERSION or raw.get("contract_version") != CONTRACT_VERSION:
        raise SpecError(f"{where}: not {FEATURES_VERSION} of contract {CONTRACT_VERSION}")
    roles = tuple(_obj(raw.get("roles"), f"{where}.roles"))
    if roles != ROLES:
        raise SpecError(f"{where}: roles {roles!r} are not {ROLES!r}")
    offsets = _obj(raw.get("offsets"), f"{where}.offsets")
    over = _obj(offsets.get("override_10hz"), f"{where}.offsets.override_10hz")
    override = Override(
        decision_hz=_number(over, "decision_hz", "override_10hz"),
        physics_steps_per_decision=_int(over, "physics_steps_per_decision", "override_10hz"),
        shapes=tuple(
            (name, _shape(s, f"override_10hz.shapes.{name}"))
            for name, s in _obj(over.get("shapes"), "override_10hz.shapes").items()
        ),
    )
    forbidden = _obj(raw.get("c24_forbidden_inputs"), f"{where}.c24_forbidden_inputs")
    names = _list(forbidden.get("names"), "c24_forbidden_inputs.names")
    patterns = _list(forbidden.get("patterns"), "c24_forbidden_inputs.patterns")
    if not all(isinstance(n, str) for n in names + patterns):
        raise SpecError(f"{where}: c24_forbidden_inputs holds a non-string")
    rules = _list(raw.get("rules"), f"{where}.rules")
    return FeatureSpec(
        path=path.resolve(),
        sha256=sha256,
        features_version=FEATURES_VERSION,
        contract_version=CONTRACT_VERSION,
        decision_hz=_number(offsets, "decision_hz", "offsets"),
        physics_steps_per_decision=_int(offsets, "physics_steps_per_decision", "offsets"),
        step_dt_s=_number(offsets, "step_dt_s", "offsets"),
        override=override,
        rules=tuple(str(r) for r in rules),
        c24_names=frozenset(str(n) for n in names),
        c24_patterns=tuple(str(p) for p in patterns),
        tensors=tuple(_tensor(t, f"{where}.tensors") for t in _list(raw.get("tensors"), f"{where}.tensors")),
    )
