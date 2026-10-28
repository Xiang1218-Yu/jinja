"""Tests for the opt-in template context contract feature."""

from __future__ import annotations

import asyncio

import pytest

from jinja2 import ContextContract
from jinja2 import ContractField as Field
from jinja2 import ContractValidationError
from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import StrictUndefined
from jinja2 import Template
from jinja2 import TemplateSyntaxError
from jinja2 import UnknownFields
from jinja2.exceptions import TemplateError
from jinja2.exceptions import UndefinedError


def contract(**fields):
    return ContextContract(fields)


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------


def test_feature_disabled_by_default_keeps_undefined_behavior(env):
    template = env.from_string("{{ a }}")

    # Default Undefined renders missing values as empty strings.
    assert template.render() == ""
    assert template.render(a={"x": 1}) == "{'x': 1}"


def test_feature_disabled_strict_undefined_unchanged():
    env = Environment(undefined=StrictUndefined)
    template = env.from_string("{{ a }}")

    with pytest.raises(UndefinedError):
        template.render()


def test_contract_fast_path_preserves_mapping_identity(env):
    template = env.from_string("{{ a }}")
    data = {"a": 1}

    # No contract anywhere: the freshly built dict is passed straight
    # through to new_context.
    assert template._prepare_context_vars(data) is data


def test_existing_exception_types_unaffected(env):
    template = env.from_string("{{ a + 1 }}")

    with pytest.raises(UndefinedError):
        template.render()


# ---------------------------------------------------------------------------
# Required / optional / defaults / basic types
# ---------------------------------------------------------------------------


def test_required_missing_raises_structured_error(env):
    template = env.from_string("{{ name }}", context_contract=contract(name=str))

    with pytest.raises(ContractValidationError) as exc_info:
        template.render()

    error = exc_info.value
    assert error.template is None  # unnamed in-memory template
    assert len(error.violations) == 1

    violation = error.violations[0]
    assert violation.path == "name"
    assert violation.reason == "missing"
    assert violation.actual_type is None
    assert violation.expected_type is None
    assert "name" in str(error)


def test_named_template_error_carries_template_name(env):
    env.loader = DictLoader({"page.html": "{{ name }}"})
    env.set_context_contract("page.html", contract(name=str))

    with pytest.raises(ContractValidationError) as exc_info:
        env.get_template("page.html").render()

    assert exc_info.value.template == "page.html"
    assert exc_info.value.violations[0].template == "page.html"


def test_all_violations_are_collected_not_fail_fast(env):
    template = env.from_string(
        "{{ a }}{{ b }}{{ c }}",
        context_contract=ContextContract(
            {"a": str, "b": int, "c": Field(list, required=False)},
        ),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(b="wrong")

    paths = {v.path: v for v in exc_info.value.violations}
    assert set(paths) == {"a", "b"}
    assert paths["b"].reason == "type"
    assert paths["b"].actual_type == "str"
    assert paths["b"].expected_type == "int"


def test_tuple_of_types_accepted(env):
    template = env.from_string("{{ n }}", context_contract=contract(n=(int, float)))
    assert template.render(n=1) == "1"
    assert template.render(n=1.5) == "1.5"

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(n="x")
    assert exc_info.value.violations[0].expected_type == "int or float"


def test_none_is_a_value_and_type_checked(env):
    template = env.from_string("{{ name }}", context_contract=contract(name=str))

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(name=None)
    violation = exc_info.value.violations[0]
    assert violation.reason == "type"
    assert violation.actual_type == "NoneType"


def test_optional_without_default_is_omitted(env):
    template = env.from_string(
        "[{{ maybe }}]",
        context_contract=contract(maybe=Field(str, required=False)),
    )
    assert template.render() == "[]"
    assert template.render(maybe="yes") == "[yes]"


def test_default_value_is_used_when_absent(env):
    template = env.from_string(
        "{{ greeting }} {{ count }}",
        context_contract=ContextContract(
            {
                "greeting": Field(str, required=False, default="hi"),
                "count": Field(int, required=False, default=3),
            }
        ),
    )
    assert template.render() == "hi 3"
    assert template.render(greeting="yo") == "yo 3"
    assert template.render(greeting="yo", count=5) == "yo 5"


def test_mutable_defaults_are_deep_copied_between_renders(env):
    template = env.from_string(
        "{% for item in items %}{{ item }}{% endfor %}",
        context_contract=contract(items=Field(list, required=False, default=[])),
    )

    assert template.render() == ""
    # A mutable default shared between renders would accumulate here.
    assert template.render(items=["a"]) == "a"
    assert template.render() == ""


def test_globals_can_satisfy_required_fields(env):
    env.globals["site"] = "example"
    template = env.from_string(
        "{{ site }}:{{ page }}", context_contract=contract(site=str, page=str)
    )
    assert template.render(page="home") == "example:home"

    # Unknown globals are never reported for a global-backed field.
    template = env.from_string(
        "{{ site }}",
        context_contract=ContextContract({"site": str}, unknown=UnknownFields.FORBID),
    )
    assert template.render() == "example"


def test_template_globals_parameter_counts(env):
    template = env.from_string(
        "{{ g }}",
        globals={"g": 9},
        context_contract=contract(g=int),
    )
    assert template.render() == "9"


# ---------------------------------------------------------------------------
# Nested objects
# ---------------------------------------------------------------------------


def test_nested_object_required_and_types(env):
    template = env.from_string(
        "{{ user.name }}/{{ user.age }}",
        context_contract=contract(
            user=Field.object(
                name=str,
                age=Field(int, required=False, default=0),
            )
        ),
    )

    assert template.render(user={"name": "a"}) == "a/0"

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(user={"age": "old"})

    by_path = {v.path: v for v in exc_info.value.violations}
    assert by_path["user.name"].reason == "missing"
    assert by_path["user.age"].reason == "type"
    assert by_path["user.age"].actual_type == "str"
    assert by_path["user.age"].expected_type == "int"


def test_nested_object_mapping_notation(env):
    # Plain nested mapping shorthand.
    template = env.from_string(
        "{{ user.name }}", context_contract=contract(user={"name": str})
    )
    assert template.render(user={"name": "a"}) == "a"


def test_nested_object_wrong_container_type(env):
    template = env.from_string(
        "{{ user.name }}",
        context_contract=contract(user=Field.object(name=str)),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(user="not-an-object")
    violation = exc_info.value.violations[0]
    assert violation.path == "user"
    assert violation.reason == "type"
    assert "object" in (violation.expected_type or "")


def test_nested_object_attribute_object(env):
    class Address:
        city = "Berlin"

    class User:
        name = "Bob"
        address = Address()

    template = env.from_string(
        "{{ user.name }}/{{ user.address.city }}",
        context_contract=contract(
            user=Field.object(
                name=str,
                address=Field.object(city=str, zip_code=Field(str, required=False)),
            )
        ),
    )
    assert template.render(user=User()) == "Bob/Berlin"

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(user=object())
    assert exc_info.value.violations[0].path == "user.name"


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------


def test_list_element_types(env):
    template = env.from_string(
        "{{ tags|join(',') }}", context_contract=contract(tags=Field.list(str))
    )
    assert template.render(tags=["a", "b"]) == "a,b"
    assert template.render(tags=("a",)) == "a"
    assert template.render(tags={"x"}) == "x"

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(tags=["ok", 3, "ok2", 4])

    by_path = {v.path: v for v in exc_info.value.violations}
    assert by_path["tags[1]"].actual_type == "int"
    assert by_path["tags[3]"].actual_type == "int"


def test_list_of_nested_objects(env):
    template = env.from_string(
        "{% for item in items %}{{ item.name }};{% endfor %}",
        context_contract=contract(
            items=Field.list(Field.object(name=str, price=float))
        ),
    )

    assert template.render(items=[{"name": "a", "price": 1.0}]) == "a;"

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(items=[{"name": "a", "price": 1.0}, {"price": 2.0}])

    by_path = {v.path: v for v in exc_info.value.violations}
    assert "items[1].name" in by_path
    assert by_path["items[1].name"].reason == "missing"


def test_list_requires_sequence(env):
    template = env.from_string(
        "{% for x in xs %}{{ x }}{% endfor %}",
        context_contract=contract(xs=Field.list(int)),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(xs=123)
    assert exc_info.value.violations[0].path == "xs"

    # Strings are not treated as element lists.
    with pytest.raises(ContractValidationError) as exc_info:
        template.render(xs="abc")
    assert exc_info.value.violations[0].path == "xs"


def test_nested_list_dimensions(env):
    template = env.from_string(
        "{{ matrix[0][0] }}",
        context_contract=contract(matrix=Field.list(Field.list(int))),
    )
    assert template.render(matrix=[[1]]) == "1"

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(matrix=[["bad"]])
    assert exc_info.value.violations[0].path == "matrix[0][0]"


# ---------------------------------------------------------------------------
# Unknown field policies
# ---------------------------------------------------------------------------


def test_unknown_fields_allow_by_default(env):
    template = env.from_string(
        "{{ known }}{{ extra }}",
        context_contract=contract(known=str),
    )
    assert template.render(known="k", extra="e") == "ke"


def test_unknown_fields_forbid(env):
    template = env.from_string(
        "{{ known }}",
        context_contract=ContextContract({"known": str}, unknown=UnknownFields.FORBID),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(known="k", extra=1, other=2)

    by_path = {v.path: v for v in exc_info.value.violations}
    assert set(by_path) == {"extra", "other"}
    assert all(v.reason == "unknown" for v in by_path.values())
    assert by_path["extra"].actual_type == "int"


def test_unknown_fields_strip_does_not_mutate_input(env):
    template = env.from_string(
        "{{ known }}/{{ extra|default('gone') }}",
        context_contract=ContextContract({"known": str}, unknown=UnknownFields.STRIP),
    )
    data = {"known": "k", "extra": "e"}

    assert template.render(data) == "k/gone"
    # Original mapping is untouched.
    assert data == {"known": "k", "extra": "e"}


def test_nested_unknown_policy_override(env):
    template = env.from_string(
        "{{ user.name }}",
        context_contract=contract(
            user=Field.object(
                {"name": str},
                unknown=UnknownFields.FORBID,
            )
        ),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(user={"name": "a", "secret": 1})
    paths = {v.path for v in exc_info.value.violations}
    assert paths == {"user.secret"}


# ---------------------------------------------------------------------------
# No mutation of caller data
# ---------------------------------------------------------------------------


def test_caller_mapping_is_never_mutated(env):
    nested = {"name": "a"}
    data = {"user": nested, "title": "t"}

    template = env.from_string(
        "{{ title }}/{{ user.name }}/{{ user.age }}",
        context_contract=contract(
            title=str,
            user=Field.object(name=str, age=Field(int, required=False, default=1)),
        ),
    )
    template.render(data)

    assert data == {"user": {"name": "a"}, "title": "t"}
    assert "age" not in nested


def test_list_elements_are_not_mutated(env):
    items = [{"name": "a"}]
    template = env.from_string(
        "{% for item in items %}{{ item.name }}/{{ item.v }}{% endfor %}",
        context_contract=contract(
            items=Field.list(
                Field.object(name=str, v=Field(int, required=False, default=0))
            )
        ),
    )
    assert template.render({"items": items}) == "a/0"
    assert items == [{"name": "a"}]


# ---------------------------------------------------------------------------
# Validation runs before any template code, repeated renders are independent
# ---------------------------------------------------------------------------


def test_no_template_code_runs_and_no_partial_output(env):
    calls = []

    def side_effect():
        calls.append(1)
        return "SIDE"

    env.globals["side_effect"] = side_effect
    template = env.from_string(
        "BEFORE{{ side_effect() }}AFTER",
        context_contract=contract(required_value=int),
    )

    with pytest.raises(ContractValidationError):
        template.render()

    assert calls == []


def test_generate_blocks_before_first_chunk(env):
    template = env.from_string("A{{ value }}B", context_contract=contract(value=int))
    iterator = iter(template.generate())

    with pytest.raises(ContractValidationError):
        next(iterator)


def test_stream_blocks_before_first_chunk(env):
    template = env.from_string("A{{ value }}B", context_contract=contract(value=int))
    stream = template.stream()  # generator created, nothing executed yet

    with pytest.raises(ContractValidationError):
        next(stream)  # consuming triggers validation before any output


def test_repeated_renders_do_not_reuse_validation_state(env):
    template = env.from_string("{{ n }}", context_contract=contract(n=int))

    for _ in range(3):
        with pytest.raises(ContractValidationError):
            template.render(n="bad")
        assert template.render(n=1) == "1"


def test_contract_validation_error_is_a_template_error(env):
    template = env.from_string("{{ a }}", context_contract=contract(a=int))

    with pytest.raises(TemplateError):
        template.render()


def test_public_validate_api(env):
    c = contract(a=int, b=Field(str, required=False))

    assert c.validate({"a": 1}) == []
    violations = c.validate({"a": "x", "c": 1})
    assert {v.path for v in violations} == {"a"}

    with pytest.raises(ContractValidationError):
        c.check({"a": "x"})


def test_bad_contract_declarations(env):
    with pytest.raises(TypeError):
        ContextContract({"a": 1})  # not a valid shorthand

    with pytest.raises(ValueError):
        ContextContract(unknown="nope")

    with pytest.raises(ValueError):
        Field(type=str, fields={"x": Field(int)})

    with pytest.raises(ValueError):
        Field(type=str, element=Field(int))

    with pytest.raises(TypeError):
        ContextContract({"a": int}).validate([("a", 1)])


# ---------------------------------------------------------------------------
# Attachment points
# ---------------------------------------------------------------------------


def test_direct_template_constructor_contract():
    template = Template("{{ a }}", context_contract=contract(a=int))
    assert template.render(a=1) == "1"

    with pytest.raises(ContractValidationError):
        template.render(a="x")


def test_environment_default_contract_for_entries():
    env = Environment(context_contract=contract(a=int))
    template = env.from_string("{{ a }}")

    assert template.render(a=2) == "2"
    with pytest.raises(ContractValidationError):
        template.render(a="x")


def test_template_contract_overrides_environment_default():
    env = Environment(context_contract=contract(a=int))
    template = env.from_string("{{ b }}", context_contract=contract(b=str))

    # The template's own contract governs; 'a' is not required.
    assert template.render(b="x", a=1) == "x"
    with pytest.raises(ContractValidationError):
        template.render(b=1)


def test_registry_get_set_remove(env):
    env.loader = DictLoader({"p.html": "{{ a }}"})

    assert env.get_context_contract("p.html") is None
    env.set_context_contract("p.html", contract(a=int))
    assert env.get_context_contract("p.html") is not None

    template = env.get_template("p.html")
    assert template.render(a=1) == "1"
    with pytest.raises(ContractValidationError):
        template.render()

    env.set_context_contract("p.html", None)
    assert template.render() == ""  # back to default undefined behavior


def test_overlay_environment_has_independent_registry():
    env = Environment(context_contract=contract(a=int))
    env.set_context_contract("p.html", contract(a=int))
    overlay = env.overlay()

    overlay.set_context_contract("q.html", contract(b=str))
    assert "q.html" not in env.context_contracts
    assert "p.html" in overlay.context_contracts
    assert overlay.context_contract is env.context_contract


def test_environment_default_contract_not_applied_to_imports():
    # The default only governs the render entry template. An imported
    # helper template has no own contract and must not be required to
    # satisfy the entry contract against globals.
    loader = DictLoader({"helpers.html": "{% macro hi(x) %}hi {{ x }}{% endmacro %}"})
    env = Environment(loader=loader, context_contract=contract(page_title=str))

    template = env.from_string(
        '{% from "helpers.html" import hi %}{{ hi(page_title) }}'
    )
    assert template.render(page_title="Home") == "hi Home"


# ---------------------------------------------------------------------------
# extends / include / import / macros
# ---------------------------------------------------------------------------


@pytest.fixture
def inheritance_env():
    loader = DictLoader(
        {
            "base.html": (
                "[base {{ title }} g={{ g }}]{% block content %}default{% endblock %}"
            ),
            "partial.html": "partial:{{ need }}",
            "partial_opt.html": "opt:{{ maybe }}",
            "macrolib.html": (
                "{% macro greet(name) %}Hello {{ name }}/{{ site }}{% endmacro %}"
            ),
            "missing_aware.html": '{% include "does_not_exist.html" ignore missing %}',
        }
    )
    env = Environment(loader=loader)
    env.set_context_contract(
        "base.html",
        ContextContract(
            {
                "title": str,
                "g": Field(int, required=False, default=0),
            }
        ),
    )
    env.set_context_contract("partial.html", contract(need=str))
    env.set_context_contract(
        "partial_opt.html", contract(maybe=Field(str, required=False))
    )
    env.set_context_contract("macrolib.html", contract(site=str))
    return env


def test_extends_merges_parent_contract(inheritance_env):
    template = inheritance_env.from_string(
        '{% extends "base.html" %}'
        "{% block content %}child {{ subtitle }}{% endblock %}",
        context_contract=contract(subtitle=str),
    )

    # Parent's 'title' is required, parent default for 'g' is applied.
    with pytest.raises(ContractValidationError) as exc_info:
        template.render(subtitle="s")
    by_path = {v.path: v for v in exc_info.value.violations}
    assert "title" in by_path
    assert by_path["title"].template == "base.html"

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(title="T")
    assert "subtitle" in {v.path for v in exc_info.value.violations}

    rendered = template.render(title="T", subtitle="S")
    assert rendered == "[base T g=0]child S"


def test_extends_strictest_unknown_policy(inheritance_env):
    inheritance_env.set_context_contract(
        "base.html",
        ContextContract({"title": str}, unknown=UnknownFields.FORBID),
    )
    template = inheritance_env.from_string(
        '{% extends "base.html" %}',
        context_contract=ContextContract({}, unknown=UnknownFields.ALLOW),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(title="t", surprise=1)
    assert {v.path for v in exc_info.value.violations} == {"surprise"}


def test_include_with_context_merges_contract(inheritance_env):
    template = inheritance_env.from_string(
        '{% include "partial.html" %}',
        context_contract=contract(header=str),
    )

    # Both the entry's 'header' and the included template's 'need' are
    # required in the initial render context.
    with pytest.raises(ContractValidationError) as exc_info:
        template.render()
    by_path = {v.path: v for v in exc_info.value.violations}
    assert by_path["need"].template == "partial.html"
    assert "header" in by_path

    rendered = template.render(header="h", need="n")
    assert rendered == "partial:n"


def test_include_without_context_validated_against_globals(inheritance_env):
    inheritance_env.globals["need"] = "from-globals"
    template = inheritance_env.from_string(
        '{% include "partial.html" without context %}',
        context_contract=contract(header=str),
    )

    # Caller context requirement still enforced.
    with pytest.raises(ContractValidationError) as exc_info:
        template.render()
    assert {v.path for v in exc_info.value.violations} == {"header"}

    # The included template renders with globals only; caller variables
    # never leak into it.
    rendered = template.render(header="h", need="caller-value")
    assert "from-globals" in rendered
    assert "caller-value" not in rendered


def test_include_without_context_missing_global_is_violation(inheritance_env):
    template = inheritance_env.from_string(
        '{% include "partial.html" without context %}',
        context_contract=contract(),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        template.render()
    violation = exc_info.value.violations[0]
    assert violation.path == "need"
    assert violation.template == "partial.html"
    assert violation.reason == "missing"


def test_import_without_context_uses_globals(inheritance_env):
    # macrolib requires 'site'; provide it as a global.
    inheritance_env.globals["site"] = "ACME"
    template = inheritance_env.from_string(
        '{% from "macrolib.html" import greet %}{{ greet(name) }}',
        context_contract=contract(name=str),
    )
    assert template.render(name="Bob") == "Hello Bob/ACME"


def test_import_without_context_does_not_see_caller_vars(inheritance_env):
    # 'site' is neither global nor on the imported template; providing it
    # in the caller context must not satisfy the import's contract.
    template = inheritance_env.from_string(
        '{% from "macrolib.html" import greet %}',
        context_contract=contract(),
    )

    with pytest.raises(ContractValidationError) as exc_info:
        template.render(site="caller")
    violation = exc_info.value.violations[0]
    assert violation.path == "site"
    assert violation.template == "macrolib.html"


def test_import_with_context_merges_contract(inheritance_env):
    template = inheritance_env.from_string(
        '{% from "partial_opt.html" import maybe with context %}',
        context_contract=contract(),
    )
    # partial_opt's 'maybe' is an optional string; the import output is
    # captured in the module and not emitted.
    assert template.render(maybe="m") == ""

    # Its type requirement still merges into the render context.
    with pytest.raises(ContractValidationError) as exc_info:
        template.render(maybe=1)
    assert exc_info.value.violations[0].path == "maybe"


def test_macro_arguments_are_not_context_contract_checked(inheritance_env):
    # A macro argument is explicit at the call site; the contract only
    # governs render context variables.
    inheritance_env.globals["site"] = "ACME"
    template = inheritance_env.from_string(
        '{% from "macrolib.html" import greet %}{{ greet(42) }}',
        context_contract=contract(),
    )
    # Passing a non-string macro argument does not raise a contract
    # error even though the macro prints it as text.
    out = template.render()
    assert out == "Hello 42/ACME"


def test_macro_free_variable_covered_by_defining_contract(inheritance_env):
    # macrolib declares 'site'; without it the render is rejected before
    # the macro ever runs.
    template = inheritance_env.from_string(
        '{% from "macrolib.html" import greet %}{{ greet("x") }}',
        context_contract=contract(),
    )
    with pytest.raises(ContractValidationError) as exc_info:
        template.render()
    assert exc_info.value.violations[0].path == "site"


def test_ignore_missing_include_contract_resolution(inheritance_env):
    template = inheritance_env.from_string(
        '{% include "missing_aware.html" %}',
        context_contract=contract(),
    )
    # No TemplateNotFound: the missing include is ignored at resolution
    # time just like at render time.
    assert template.render() == ""


def test_incompatible_parent_child_contracts_raise(inheritance_env):
    inheritance_env.set_context_contract("base.html", contract(title=int))
    template = inheritance_env.from_string(
        '{% extends "base.html" %}',
        context_contract=contract(title=str),
    )

    with pytest.raises(ValueError) as exc_info:
        template.render(title="x")
    assert "title" in str(exc_info.value)


def test_transitive_contract_chain(inheritance_env):
    # base.html includes partial.html transitively would be unusual, so
    # build: child extends base; base extends is impossible twice; use
    # child includes base-like chain via two includes.
    loader = DictLoader(
        {
            "a.html": 'A{% include "b.html" %}',
            "b.html": 'B{% include "c.html" %}',
            "c.html": "C{{ deep }}",
        }
    )
    env = Environment(loader=loader)
    env.set_context_contract("c.html", contract(deep=str))

    template = env.from_string('{% include "a.html" %}', context_contract=contract())

    with pytest.raises(ContractValidationError) as exc_info:
        template.render()
    assert exc_info.value.violations[0].path == "deep"
    assert exc_info.value.violations[0].template == "c.html"

    assert template.render(deep="down") == "ABCdown"


def test_dynamic_include_is_not_statically_resolved(inheritance_env):
    template = inheritance_env.from_string(
        "{% set tpl = 'partial.html' %}{% include tpl %}",
        context_contract=contract(),
    )
    # Dynamic references cannot be resolved up front; rendering falls
    # back to normal runtime behavior (and 'need' renders empty).
    assert template.render() == "partial:"


def test_contract_refs_collected_statically(env):
    template = env.from_string(
        '{% extends "base.html" %}'
        '{% include "inc.html" %}'
        '{% import "lib.html" as lib %}'
        '{% from "f.html" import x %}'
    )
    kinds = {(kind, name) for kind, name, _, _, _ in template._contract_refs}
    assert ("extends", "base.html") in kinds
    assert ("include", "inc.html") in kinds
    assert ("import", "lib.html") in kinds
    assert ("from_import", "f.html") in kinds


def test_without_context_flag_present_in_refs(env):
    template = env.from_string(
        '{% include "a.html" %}{% include "b.html" without context %}'
    )
    flags = {name: with_ctx for _, name, with_ctx, _, _ in template._contract_refs}
    assert flags["a.html"] is True
    assert flags["b.html"] is False


def test_refs_mark_references_nested_in_local_scopes(env):
    template = env.from_string(
        '{% include "top.html" %}'
        '{% for x in xs %}{% include "loop.html" %}{% endfor %}'
        '{% macro m() %}{% include "macro.html" %}{% endmacro %}'
    )
    scoped = {
        name: local_scope for _, name, _, _, local_scope in template._contract_refs
    }
    assert scoped["top.html"] is False
    assert scoped["loop.html"] is True
    assert scoped["macro.html"] is True


def test_include_inside_loop_not_required_in_initial_context(inheritance_env):
    # The partial requires 'need', but inside the loop it is provided by
    # the loop variable; that is a local value like a macro argument, not
    # part of the render-entry context.
    template = inheritance_env.from_string(
        '{% for need in needs %}{% include "partial.html" %}{% endfor %}',
        context_contract=contract(needs=Field.list(str)),
    )
    assert template.render(needs=["a", "b"]) == "partial:apartial:b"

    # Rendering the partial directly as an entry still enforces its
    # contract against the initial context.
    direct = inheritance_env.get_template("partial.html")
    with pytest.raises(ContractValidationError):
        direct.render()


def test_multi_candidate_include_merges_all_contracts():
    loader = DictLoader(
        {
            "a.html": "A{{ a_need }}",
            "b.html": "B{{ b_need }}",
        }
    )
    env = Environment(loader=loader)
    env.set_context_contract("a.html", contract(a_need=str))
    env.set_context_contract("b.html", contract(b_need=str))

    template = env.from_string(
        '{% include ["a.html", "b.html"] %}', context_contract=contract()
    )

    # Every candidate that could be selected is required statically.
    with pytest.raises(ContractValidationError) as exc_info:
        template.render(a_need="x")
    paths = {v.path for v in exc_info.value.violations}
    assert paths == {"b_need"}

    assert template.render(a_need="x", b_need="y") in ("Ax", "By")


def test_missing_nested_object_uses_default_copy(env):
    template = env.from_string(
        "{% if user %}{{ user.name }}{% endif %}",
        context_contract=contract(
            user=Field.object({"name": str}, required=False, default={"name": "nobody"})
        ),
    )
    assert template.render() == "nobody"


def test_public_validate_stamps_template_name(env):
    violations = ContextContract({"a": int}).validate(
        {"a": "x"}, template_name="mail.html"
    )
    assert violations[0].template == "mail.html"


def test_contract_works_with_sandboxed_environment():
    from jinja2.sandbox import SandboxedEnvironment

    env = SandboxedEnvironment()
    template = env.from_string("{{ a }}", context_contract=contract(a=int))
    assert template.render(a=3) == "3"

    with pytest.raises(ContractValidationError):
        template.render(a="x")


# ---------------------------------------------------------------------------
# Async rendering
# ---------------------------------------------------------------------------


def test_async_render_validates_before_execution():
    loader = DictLoader({"page.html": "{{ value }}"})
    env = Environment(loader=loader, enable_async=True)
    env.set_context_contract("page.html", contract(value=int))
    template = env.get_template("page.html")

    async def main():
        with pytest.raises(ContractValidationError):
            await template.render_async(value="bad")
        return await template.render_async(value=42)

    assert asyncio.run(main()) == "42"


def test_async_generate_blocks_before_output():
    env = Environment(enable_async=True)
    template = env.from_string("{{ value }}", context_contract=contract(value=int))

    async def main():
        agen = template.generate_async()
        with pytest.raises(ContractValidationError):
            await agen.__anext__()

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def test_native_environment_contract():
    from jinja2.nativetypes import NativeEnvironment

    env = NativeEnvironment()
    template = env.from_string("{{ a + 1 }}", context_contract=contract(a=int))
    assert template.render(a=1) == 2

    with pytest.raises(ContractValidationError):
        template.render(a="1")


def test_positional_mapping_argument(env):
    template = env.from_string("{{ a }}", context_contract=contract(a=int))
    assert template.render({"a": 5}) == "5"

    with pytest.raises(ContractValidationError):
        template.render({"a": "x"})


def test_syntax_errors_still_surface_before_contracts(env):
    with pytest.raises(TemplateSyntaxError):
        env.from_string("{% for x in a %}", context_contract=contract(a=list))
