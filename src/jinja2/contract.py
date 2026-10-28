"""Template context contracts.

A context contract is an opt-in declaration of the variables a template
expects when it is rendered: required and optional fields, default
values, basic types, nested objects, list element types and a policy for
unknown fields. When a contract is in effect, the whole context is
validated *before* :meth:`Template.render` or
:meth:`Template.render_async` starts executing template code. If
validation fails, a :class:`~jinja2.exceptions.ContractValidationError`
is raised, no template code runs and no partial output is produced.

The feature is disabled by default. Templates without a contract keep
the existing :class:`~jinja2.runtime.Context`, :class:`Undefined` and
exception behavior without any validation overhead.

Inheritance rules across templates (resolved statically before
rendering):

* ``{% extends %}`` always merges the parent contract into the child's
  effective contract; the parent renders with the same context.
* Top level ``{% include %}`` and ``{% import %}`` with context merge
  their contracts into the render-entry contract.
* ``{% include ... without context %}`` and plain imports (which are
  without context by default) only see globals, so their contracts are
  validated independently against the environment and template globals.
* References nested in a ``for`` loop, macro or ``call`` scope are not
  merged: their required fields may be supplied by loop targets or
  macro/call arguments at runtime (which, like macro arguments, are not
  part of the render-entry context contract).

Dynamic template references (variable template names) cannot be
resolved before rendering and are ignored by static resolution.

.. versionadded:: 3.2
"""

from __future__ import annotations

import copy
import typing as t
from collections import abc
from dataclasses import dataclass
from dataclasses import replace

from .exceptions import ContractValidationError
from .exceptions import TemplateNotFound

if t.TYPE_CHECKING:
    from .environment import Environment
    from .environment import Template

__all__ = [
    "UnknownFields",
    "ContractViolation",
    "ContractField",
    "ContextContract",
    "prepare_render_context",
]


class UnknownFields:
    """Policies for fields that are not declared by a contract.

    ``ALLOW`` (the default) keeps unknown fields and passes them to the
    template unchanged. ``FORBID`` turns every unknown field into a
    violation. ``STRIP`` removes unknown fields from the (copied) context
    before rendering without touching the caller's mapping.
    """

    ALLOW = "allow"
    FORBID = "forbid"
    STRIP = "strip"


_UNKNOWN_POLICIES = frozenset(
    {UnknownFields.ALLOW, UnknownFields.FORBID, UnknownFields.STRIP}
)
#: Strictest policy wins when contracts are inherited across templates.
_POLICY_RANK = {
    UnknownFields.ALLOW: 0,
    UnknownFields.STRIP: 1,
    UnknownFields.FORBID: 2,
}

_UNSET: t.Any = object()

_REASON_MISSING = "missing"
_REASON_TYPE = "type"
_REASON_UNKNOWN = "unknown"

#: Concrete sequence types checked for list element contracts. Strings are
#: deliberately excluded even though they are sequences.
_LIST_TYPES = (list, tuple, set, frozenset)

#: Built-in value types that can never satisfy a nested object contract.
#: Anything else (dataclasses, ORM models, modules, ...) is validated
#: through read-only attribute access.
_SCALAR_TYPES = (
    str,
    bytes,
    bytearray,
    memoryview,
    int,
    float,
    complex,
    bool,
    range,
    type(None),
) + _LIST_TYPES


@dataclass(frozen=True)
class ContractViolation:
    """A single contract violation.

    :param path: Dotted field path, with list indices such as
        ``user.address.city`` or ``items[0].name``.
    :param reason: One of ``"missing"``, ``"type"`` or ``"unknown"``.
    :param template: Name of the template whose contract requires the
        field, or ``None`` for the entry template.
    :param actual_type: Name of the type of the provided value, or
        ``None`` for missing fields.
    :param expected_type: Human readable description of the expected
        type, or ``None`` for missing/unknown fields.
    """

    path: str
    reason: str
    template: str | None = None
    actual_type: str | None = None
    expected_type: str | None = None

    @property
    def message(self) -> str:
        """Human readable description of the violation."""
        where = self.path if self.path else "<root>"

        if self.reason == _REASON_MISSING:
            msg = f"field {where!r} is required but was not provided"
        elif self.reason == _REASON_UNKNOWN:
            msg = f"unknown field {where!r} is not allowed by the contract"
            if self.actual_type is not None:
                msg = f"{msg} (got {self.actual_type})"
        else:
            msg = (
                f"field {where!r} expected {self.expected_type}"
                f" but got {self.actual_type}"
            )

        if self.template is not None:
            msg = f"{msg} (required by template {self.template!r})"

        return msg


@dataclass(frozen=True)
class ContractField:
    """Declaration of a single field.

    :param type: A type or tuple of types accepted by ``isinstance``.
    :param required: Whether the field must be present.
    :param default: Value used when the field is absent. The presence of
        a default makes the field optional.
    :param fields: Mapping of nested field declarations; turns the field
        into a nested object contract.
    :param element: Declaration applied to every element of a list.
    :param unknown: Per-object override of the unknown field policy.
    """

    type: type | tuple[type, ...] | None = None
    required: bool = True
    default: t.Any = _UNSET
    fields: t.Mapping[str, ContractField] | None = None
    element: ContractField | None = None
    unknown: str | None = None

    def __post_init__(self) -> None:
        _validate_declaration(self, "")

    @property
    def has_default(self) -> bool:
        return self.default is not _UNSET

    @property
    def is_object(self) -> bool:
        return self.fields is not None

    @property
    def is_list(self) -> bool:
        return self.element is not None

    @classmethod
    def object(
        cls,
        fields: t.Mapping[str, ContractFieldLike] | None = None,
        /,
        *,
        required: bool = True,
        default: t.Any = _UNSET,
        unknown: str | None = None,
        **kwargs_fields: ContractFieldLike,
    ) -> ContractField:
        """Create a nested object field.

        Nested fields can be passed as a mapping and/or as keyword
        arguments. Keyword arguments take precedence.
        """
        merged: dict[str, ContractFieldLike] = dict(fields or {})
        merged.update(kwargs_fields)
        return cls(
            required=required,
            default=default,
            fields={name: coerce_field(spec) for name, spec in merged.items()},
            unknown=unknown,
        )

    @classmethod
    def list(
        cls,
        element: ContractFieldLike | None = None,
        /,
        *,
        required: bool = True,
        default: t.Any = _UNSET,
    ) -> ContractField:
        """Create a list field whose elements are validated by
        ``element`` (a type, nested field shorthand or
        :class:`ContractField`).
        """
        return cls(
            required=required,
            default=default,
            element=coerce_field(element) if element is not None else ContractField(),
        )


ContractFieldLike = ContractField | type | tuple[t.Any, ...] | t.Mapping[str, t.Any]


def coerce_field(spec: ContractFieldLike) -> ContractField:
    """Coerce shorthand notation into a :class:`ContractField`.

    Types (and tuples of types) become plain typed fields, mappings become
    nested object fields and :class:`ContractField` instances are used as
    is.
    """
    if isinstance(spec, ContractField):
        return spec

    if isinstance(spec, type) or (
        isinstance(spec, tuple) and spec and all(isinstance(x, type) for x in spec)
    ):
        return ContractField(type=t.cast("type | tuple[type, ...]", spec))

    if isinstance(spec, abc.Mapping):
        return ContractField(
            fields={
                str(name): coerce_field(t.cast("ContractFieldLike", sub))
                for name, sub in spec.items()
            }
        )

    raise TypeError(
        "Contract field must be a type, tuple of types, mapping of nested"
        f" fields, or ContractField instance, got {spec!r}"
    )


def _validate_declaration(field: ContractField, path: str) -> None:
    if field.type is not None and not isinstance(field.type, tuple):
        if not isinstance(field.type, type):
            raise ValueError(f"{path}: type must be a type or tuple of types")
    if isinstance(field.type, tuple) and not field.type:
        raise ValueError(f"{path}: type tuple must not be empty")

    if field.fields is not None and field.type is not None:
        raise ValueError(f"{path}: cannot combine 'type' with nested 'fields'")
    if field.element is not None and field.type is not None:
        raise ValueError(f"{path}: cannot combine 'type' with list 'element'")
    if field.unknown is not None and field.unknown not in _UNKNOWN_POLICIES:
        raise ValueError(
            f"{path}: unknown field policy must be one of 'allow', 'forbid' or 'strip'"
        )

    for name, sub in (field.fields or {}).items():
        _validate_declaration(sub, f"{path}.{name}" if path else name)
    if field.element is not None:
        _validate_declaration(field.element, f"{path}[]" if path else "[]")


def _type_names(types: type | tuple[type, ...] | None) -> str | None:
    if types is None:
        return None
    if isinstance(types, tuple):
        return " or ".join(t.__name__ for t in types)
    return types.__name__


def _value_type_name(value: t.Any) -> str:
    return type(value).__name__


class _AttrView(abc.Mapping[str, t.Any]):
    """Read-only mapping adapter over a non-mapping object.

    Only declared fields are visible, so unknown-field policies cannot be
    applied to arbitrary object attributes (methods, descriptors, ...).
    Defaults are not injected either because that would require mutating
    or wrapping the user's object.
    """

    __slots__ = ("_obj", "_names")

    def __init__(self, obj: t.Any, names: t.Iterable[str]) -> None:
        self._obj = obj
        self._names = {name for name in names if hasattr(obj, name)}

    def __contains__(self, key: object) -> bool:
        return key in self._names

    def __getitem__(self, key: str) -> t.Any:
        try:
            return getattr(self._obj, key)
        except AttributeError:
            raise KeyError(key) from None

    def __iter__(self) -> t.Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


def _is_mapping(value: t.Any) -> t.TypeGuard[abc.Mapping[str, t.Any]]:
    return isinstance(value, abc.Mapping)


class ContextContract:
    """Declaration of the context expected by a template render.

    :param fields: Mapping of field names to declarations. Values may be
        :class:`ContractField` instances, plain types/tuples (shorthand
        for a typed required field) or mappings (shorthand for a nested
        object).
    :param unknown: Policy for fields not declared in ``fields``; one of
        the :class:`UnknownFields` constants.
    :param name: Optional label for error messages.
    """

    def __init__(
        self,
        fields: t.Mapping[str, ContractFieldLike] | None = None,
        *,
        unknown: str = UnknownFields.ALLOW,
        name: str | None = None,
    ) -> None:
        if unknown not in _UNKNOWN_POLICIES:
            raise ValueError(
                "unknown field policy must be one of 'allow', 'forbid' or 'strip'"
            )
        self.unknown: str = unknown
        self.name: str | None = name
        self.fields: dict[str, ContractField] = {
            str(field_name): coerce_field(spec)
            for field_name, spec in (fields or {}).items()
        }
        for field_name, spec in self.fields.items():
            _validate_declaration(spec, field_name)

    # -- construction -------------------------------------------------

    def clone(self) -> ContextContract:
        cloned = ContextContract.__new__(ContextContract)
        cloned.unknown = self.unknown
        cloned.name = self.name
        cloned.fields = dict(self.fields)
        return cloned

    def __repr__(self) -> str:
        return f"ContextContract({sorted(self.fields)!r}, unknown={self.unknown!r})"

    # -- public validation API ----------------------------------------

    def validate(
        self,
        data: abc.Mapping[str, t.Any],
        *,
        template_name: str | None = None,
    ) -> list[ContractViolation]:
        """Validate ``data`` without rendering and return all violations.

        The mapping and the objects it contains are never modified. An
        empty list means the data satisfies the contract.
        """
        if not _is_mapping(data):
            raise TypeError("contract validation expects a mapping")

        violations: list[ContractViolation] = []
        _check_mapping(
            self,
            data,
            globals_view=None,
            path="",
            stamp=template_name,
            violations=violations,
            policy=self.unknown,
            enumerate_unknown=True,
        )
        return violations

    def check(
        self,
        data: abc.Mapping[str, t.Any],
        *,
        template_name: str | None = None,
    ) -> None:
        """Validate ``data`` and raise
        :class:`~jinja2.exceptions.ContractValidationError` if it
        contains any violations.
        """
        violations = self.validate(data, template_name=template_name)

        if violations:
            raise ContractValidationError(violations, template_name)


# ---------------------------------------------------------------------------
# Recursive validation
# ---------------------------------------------------------------------------


def _effective_policy(field: ContractField, parent_policy: str) -> str:
    return field.unknown if field.unknown is not None else parent_policy


def _violation(
    violations: list[ContractViolation],
    path: str,
    reason: str,
    stamp: str | None,
    *,
    actual: t.Any = _UNSET,
    expected: str | None = None,
) -> None:
    violations.append(
        ContractViolation(
            path=path,
            reason=reason,
            template=stamp,
            actual_type=None if actual is _UNSET else _value_type_name(actual),
            expected_type=expected,
        )
    )


def _lookup(container: t.Any, key: str) -> tuple[bool, t.Any]:
    if _is_mapping(container):
        if key in container:
            return True, container[key]
        return False, _UNSET

    try:
        value = getattr(container, key)
    except AttributeError:
        return False, _UNSET
    return True, value


def _check_mapping(
    contract: ContextContract,
    data: abc.Mapping[str, t.Any],
    *,
    globals_view: abc.Mapping[str, t.Any] | None,
    path: str,
    stamp: str | None,
    violations: list[ContractViolation],
    policy: str,
    enumerate_unknown: bool,
) -> dict[str, t.Any]:
    """Validate a mapping described by ``contract``.

    Returns a new mapping with defaults applied and unknown fields
    stripped according to ``policy``. The input mapping is never
    mutated.
    """
    result: dict[str, t.Any] = {}

    # Provided fields, including undeclared ones (subject to policy).
    for key, value in data.items():
        if key in contract.fields:
            sub_path = f"{path}.{key}" if path else str(key)
            result[key] = _check_value(
                contract.fields[key],
                value,
                sub_path,
                stamp,
                violations,
                policy,
            )
        elif policy == UnknownFields.FORBID and enumerate_unknown:
            _violation(
                violations,
                f"{path}.{key}" if path else str(key),
                _REASON_UNKNOWN,
                stamp,
                actual=value,
            )
        elif policy == UnknownFields.STRIP:
            # Drop the field from the copied context.
            continue
        else:
            result[key] = value

    # Declared fields that were not provided.
    for key, field in contract.fields.items():
        if key in result:
            continue

        sub_path = f"{path}.{key}" if path else key
        present, value = (False, _UNSET)

        if key not in data and globals_view is not None:
            present, value = _lookup(globals_view, key)

        if present:
            result[key] = _check_value(
                field, value, sub_path, stamp, violations, policy
            )
        elif field.has_default:
            # Deep copy so repeated renders never share mutable defaults
            # and the caller's objects are never touched.
            result[key] = copy.deepcopy(field.default)
        elif field.required:
            _violation(violations, sub_path, _REASON_MISSING, stamp)
        # Optional without default: leave the field absent; the template
        # sees its regular Undefined behavior.

    return result


def _check_object_field(
    field: ContractField,
    value: t.Any,
    path: str,
    stamp: str | None,
    violations: list[ContractViolation],
    policy: str,
) -> t.Any:
    object_policy = _effective_policy(field, policy)

    if _is_mapping(value):
        child = ContextContract.__new__(ContextContract)
        child.unknown = object_policy
        child.name = None
        child.fields = dict(field.fields or {})
        return _check_mapping(
            child,
            value,
            globals_view=None,
            path=path,
            stamp=stamp,
            violations=violations,
            policy=object_policy,
            enumerate_unknown=True,
        )

    # Support attribute objects (dataclasses, ORM models, ...) on a
    # read-only basis. Scalar/builtin types cannot satisfy an object
    # contract.
    if field.fields is not None and not isinstance(value, _SCALAR_TYPES):
        view = _AttrView(value, field.fields)
        child = ContextContract.__new__(ContextContract)
        child.unknown = UnknownFields.ALLOW
        child.name = None
        child.fields = dict(field.fields)
        # Results cannot be written back without mutating the object, so
        # discard the prepared mapping.
        _check_mapping(
            child,
            view,
            globals_view=None,
            path=path,
            stamp=stamp,
            violations=violations,
            policy=UnknownFields.ALLOW,
            enumerate_unknown=False,
        )
        return value

    _violation(
        violations,
        path,
        _REASON_TYPE,
        stamp,
        actual=value,
        expected="a mapping or object",
    )
    return value


def _check_list_field(
    field: ContractField,
    value: t.Any,
    path: str,
    stamp: str | None,
    violations: list[ContractViolation],
    policy: str,
) -> t.Any:
    assert field.element is not None

    if not isinstance(value, _LIST_TYPES) or isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        _violation(
            violations,
            path,
            _REASON_TYPE,
            stamp,
            actual=value,
            expected="a list",
        )
        return value

    checked: list[t.Any] = []
    changed = False

    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        new_item = _check_value(
            field.element, item, item_path, stamp, violations, policy
        )
        checked.append(new_item)
        changed = changed or new_item is not item

    if not changed:
        return value

    if isinstance(value, tuple):
        return tuple(checked)
    if isinstance(value, frozenset):
        return frozenset(checked)
    if isinstance(value, set):
        return set(checked)
    return checked


def _check_value(
    field: ContractField,
    value: t.Any,
    path: str,
    stamp: str | None,
    violations: list[ContractViolation],
    policy: str,
) -> t.Any:
    if field.type is not None and not isinstance(value, field.type):
        _violation(
            violations,
            path,
            _REASON_TYPE,
            stamp,
            actual=value,
            expected=_type_names(field.type),
        )
        return value

    if field.fields is not None:
        return _check_object_field(field, value, path, stamp, violations, policy)

    if field.element is not None:
        return _check_list_field(field, value, path, stamp, violations, policy)

    return value


# ---------------------------------------------------------------------------
# Contract inheritance across extends / include / import
# ---------------------------------------------------------------------------


def _stricter(a: str, b: str) -> str:
    return a if _POLICY_RANK[a] >= _POLICY_RANK[b] else b


def _merge_field(
    a: ContractField,
    b: ContractField,
    path: str,
    a_template: str | None,
    b_template: str | None,
    added: dict[str, str],
) -> ContractField:
    # Explicit type declarations must agree.
    if a.type is not None and b.type is not None and a.type != b.type:
        raise ValueError(
            f"Incompatible context contracts for field {path!r}: declared as"
            f" {_type_names(a.type)} in template {a_template!r} but as"
            f" {_type_names(b.type)} in template {b_template!r}"
        )

    # One side declares structure the other side does not know about.
    if (a.fields is not None and b.fields is None) or (
        a.fields is None and b.fields is not None
    ):
        structured, other = (a, b) if a.fields is not None else (b, a)
        if other.type is not None:
            raise ValueError(
                f"Incompatible context contracts for field {path!r}: declared"
                f" as an object in one template but as"
                f" {_type_names(other.type)} in another"
            )
        added[path] = ""
        merged_fields = dict(structured.fields or {})
    elif a.fields is not None and b.fields is not None:
        merged_fields = dict(a.fields)
        for name, b_sub in b.fields.items():
            sub_path = f"{path}.{name}" if path else name
            if name in merged_fields:
                merged_fields[name] = _merge_field(
                    merged_fields[name],
                    b_sub,
                    sub_path,
                    a_template,
                    b_template,
                    added,
                )
            else:
                merged_fields[name] = b_sub
                added[sub_path] = ""
    else:
        merged_fields = None

    if (a.element is not None and b.element is None) or (
        a.element is None and b.element is not None
    ):
        raise ValueError(
            f"Incompatible context contracts for field {path!r}: declared as a"
            " list in one template but as a non-list in another"
        )

    merged_element: ContractField | None = None
    if a.element is not None and b.element is not None:
        merged_element = _merge_field(
            a.element,
            b.element,
            f"{path}[]",
            a_template,
            b_template,
            added,
        )

    unknown: str | None
    if a.unknown is None or b.unknown is None:
        unknown = a.unknown or b.unknown
    else:
        unknown = _stricter(a.unknown, b.unknown)

    return replace(
        a,
        type=a.type or b.type,
        required=a.required or b.required,
        default=a.default if a.has_default else b.default,
        fields=merged_fields,
        element=merged_element,
        unknown=unknown,
    )


def _merge_contracts(
    a: ContextContract,
    b: ContextContract,
    a_template: str | None,
    b_template: str | None,
    b_provenance: t.Mapping[str, str],
) -> tuple[ContextContract, dict[str, str]]:
    """Merge ``b`` into ``a`` (both share the same render context).

    Returns the merged contract and a mapping of field paths contributed
    by ``b`` (or its own dependencies) to their originating templates.
    """
    merged = a.clone()
    added: dict[str, str] = {}

    for name, b_field in b.fields.items():
        if name in merged.fields:
            merged.fields[name] = _merge_field(
                merged.fields[name],
                b_field,
                name,
                a_template,
                b_template,
                added,
            )
        else:
            merged.fields[name] = b_field
            added[name] = ""

    merged.unknown = _stricter(a.unknown, b.unknown)

    # Every path contributed by this merge is attributed to the template
    # that actually required it, which may be a transitive dependency.
    origins = {path: b_provenance.get(path) or b_template or "" for path in added}
    return merged, origins


class _ContractResolver:
    """Resolves the effective contract of one render by walking the
    statically known extends/include/import references of templates.
    """

    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self.memo: dict[
            int,
            tuple[ContextContract, dict[str, str], list[ContractViolation]],
        ] = {}

    def _own_contract(
        self, template: Template, is_entry: bool
    ) -> ContextContract | None:
        contract = template.context_contract

        if contract is None and template.name is not None:
            contract = self.environment.context_contracts.get(template.name)

        # The environment-wide default contract only applies to the
        # template that rendering starts on; it must not suddenly apply
        # to imported helper templates that only see globals.
        if contract is None and is_entry:
            contract = self.environment.context_contract

        return contract

    def resolve(
        self, template: Template, *, is_entry: bool = False
    ) -> tuple[ContextContract, dict[str, str], list[ContractViolation]]:
        key = id(template)
        cached = self.memo.get(key)

        if cached is not None:
            return cached

        own = self._own_contract(template, is_entry)
        merged = own.clone() if own is not None else ContextContract()
        provenance: dict[str, str] = {}
        standalone_violations: list[ContractViolation] = []

        refs = t.cast(
            t.Iterable[tuple[str, str, bool, bool, bool]],
            getattr(template, "_contract_refs", ()),
        )

        for kind, ref_name, with_context, ignore_missing, local_scope in refs:
            try:
                dependency = self.environment.get_template(
                    ref_name, parent=template.name
                )
            except TemplateNotFound:
                if kind == "include" and ignore_missing:
                    continue
                raise

            dep_contract, dep_provenance, transitive_violations = self.resolve(
                dependency
            )
            standalone_violations.extend(transitive_violations)

            if (kind == "extends" or with_context) and not local_scope:
                # extends and top level "include/import with context"
                # execute with the caller's render context, so their
                # requirements are merged into this contract.
                merged, origins = _merge_contracts(
                    merged,
                    dep_contract,
                    template.name,
                    dependency.name,
                    dep_provenance,
                )
                provenance.update(origins)
            elif local_scope and kind != "extends":
                # Nested inside a for/macro/call scope: required fields
                # may be supplied by loop targets, macro or call
                # arguments that only exist at runtime. Like macro
                # arguments, they are not part of the render-entry
                # context contract. Without-context dependencies of this
                # template were already collected transitively.
                continue
            else:
                # "include without context" and plain imports only see
                # globals, so their contracts are validated on their own
                # against globals and never merged into the caller's.
                globals_violations: list[ContractViolation] = []
                _check_mapping(
                    dep_contract,
                    t.cast("abc.Mapping[str, t.Any]", dependency.globals),
                    globals_view=None,
                    path="",
                    stamp=None,
                    violations=globals_violations,
                    policy=dep_contract.unknown,
                    # Globals always contain engine provided names such
                    # as 'range' or 'dict', which must not be reported.
                    enumerate_unknown=False,
                )
                standalone_violations.extend(
                    _attribute(globals_violations, dep_provenance, dependency.name)
                )

        result = (merged, provenance, standalone_violations)
        self.memo[key] = result
        return result


def _attribute(
    violations: list[ContractViolation],
    provenance: t.Mapping[str, str],
    default_template: str | None,
) -> list[ContractViolation]:
    """Fill in the requiring template for every violation.

    The longest declared path prefix wins so fields added by a nested
    dependency are attributed to that dependency.
    """
    prefixes = sorted(provenance, key=len, reverse=True)
    attributed: list[ContractViolation] = []

    for violation in violations:
        origin: str | None = None

        for prefix in prefixes:
            if (
                violation.path == prefix
                or violation.path.startswith(prefix + ".")
                or violation.path.startswith(prefix + "[")
            ):
                origin = provenance[prefix] or None
                break

        attributed.append(replace(violation, template=origin or default_template))

    return attributed


def prepare_render_context(
    template: Template, vars: dict[str, t.Any]
) -> dict[str, t.Any]:
    """Validate and prepare the context variables before rendering.

    Returns a new dict with defaults applied and unknown fields handled
    according to the effective contract. Raises
    :class:`~jinja2.exceptions.ContractValidationError` containing every
    violation before any template code can run. Called internally by
    :meth:`Template.render` and friends when contracts are enabled.
    """
    environment = template.environment

    # Fast path: nothing is enabled anywhere, keep the exact existing
    # behavior and object identity.
    if (
        template.context_contract is None
        and environment.context_contract is None
        and not environment.context_contracts
    ):
        return vars

    merged, provenance, standalone_violations = _ContractResolver(environment).resolve(
        template, is_entry=True
    )

    violations: list[ContractViolation] = []
    prepared = _check_mapping(
        merged,
        vars,
        globals_view=t.cast("abc.Mapping[str, t.Any]", template.globals),
        path="",
        stamp=None,
        violations=violations,
        policy=merged.unknown,
        enumerate_unknown=True,
    )
    violations = _attribute(violations, provenance, template.name)
    violations.extend(standalone_violations)

    if violations:
        raise ContractValidationError(violations, template.name)

    return prepared
