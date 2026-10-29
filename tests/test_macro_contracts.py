import json
import pickle

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2.exceptions import MacroContractError
from jinja2.exceptions import TemplateAssertionError
from jinja2.exceptions import TemplateSyntaxError
from jinja2.runtime import CONTRACT_VERSION
from jinja2.sandbox import SandboxedEnvironment

CARD = (
    "{% macro card(title, body='-') "
    "contract {'required': ['title'], 'optional': {'body': '-'}}"
    " %}<{{ title }}|{{ body }}>{% endmacro %}"
)


def make_env(**kwargs):
    kwargs.setdefault("macro_contracts", True)
    return Environment(**kwargs)


# ---------------------------------------------------------------------------
# default off / legacy behaviour
# ---------------------------------------------------------------------------


def test_contracts_disabled_by_default(env):
    assert env.macro_contract_enabled is False
    src = (
        "{% macro f(a, b='x') contract {'required': ['a']} %}{{ a }}/{{ b }}"
        "{% endmacro %}"
    )
    # missing required argument keeps the historical Undefined behaviour
    assert env.from_string(src + "{{ f() }}").render() == "/x"
    # extra positional / keyword arguments keep raising TypeErrors
    with pytest.raises(TypeError):
        env.from_string(src + "{{ f(1, 2, 3) }}").render()
    with pytest.raises(TypeError):
        env.from_string(src + "{{ f(1, z=2) }}").render()
    # nothing is registered
    macro = env.from_string(src).module.f
    assert macro not in env.macro_contracts


def test_enabled_but_macro_without_contract_keeps_legacy_semantics():
    env = make_env()
    src = "{% macro f(a) %}[{{ a }}]{% endmacro %}"
    assert env.from_string(src + "[{{ f() }}]").render() == "[[]]"
    with pytest.raises(TypeError):
        env.from_string(src + "{{ f(1, z=2) }}").render()


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


def test_required_optional_and_defaults():
    env = make_env()
    t = env.from_string(CARD + "{{ card('A') }}|{{ card('B', body='x') }}")
    assert t.render() == "<A|->|<B|x>"


def test_derived_contract_from_signature():
    env = make_env()
    src = "{% macro f(a, b='d') contract {} %}{{ a }}/{{ b }}{% endmacro %}"
    assert env.from_string(src + "{{ f(1) }}").render() == "1/d"
    with pytest.raises(MacroContractError) as exc:
        env.from_string(src + "{{ f() }}").render()
    assert exc.value.reason == "missing_argument"
    assert exc.value.info["missing"] == ["a"]


def test_keyword_for_required_argument():
    env = make_env()
    t = env.from_string(CARD + "{{ card(title='K') }}")
    assert t.render() == "<K|->"


# ---------------------------------------------------------------------------
# structured error
# ---------------------------------------------------------------------------


def _render_error(src, call):
    env = make_env()
    with pytest.raises(MacroContractError) as exc:
        env.from_string(src + call).render()
    return exc.value


def test_missing_required_is_structured():
    err = _render_error(CARD, "{{ card() }}")
    assert err.reason == "missing_argument"
    assert err.macro_name == "card"
    assert err.contract_version == CONTRACT_VERSION
    assert err.info["missing"] == ["title"]
    assert err.call["argc"] == 0
    assert err.call["kwarg_names"] == []
    data = err.to_dict()
    # the structured payload must be json serializable
    json.dumps(data)
    assert data["macro_name"] == "card"
    assert data["contract_version"] == 1
    assert data["contract"]["required"] == ["title"]
    assert data["contract"]["optional"] == ["body"]


def test_custom_contract_version_reported():
    err = _render_error(
        "{% macro f(a) contract {'required': ['a'], 'version': 9} %}{{ a }}"
        "{% endmacro %}",
        "{{ f() }}",
    )
    assert err.contract_version == 9
    assert err.to_dict()["contract_version"] == 9


def test_too_many_positional_arguments():
    err = _render_error(CARD, "{{ card(1, 2, 3) }}")
    assert err.reason == "too_many_positional_arguments"
    assert err.info["maximum"] == 2
    assert err.info["given"] == 3
    assert err.call["argc"] == 3


def test_unexpected_keyword_argument():
    err = _render_error(CARD, "{{ card(1, bogus=2) }}")
    assert err.reason == "unexpected_keyword_argument"
    assert err.info["argument"] == "bogus"
    assert "bogus" in err.call["kwarg_names"]


def test_multiple_values_for_argument():
    err = _render_error(CARD, "{{ card(1, title=2) }}")
    assert err.reason == "multiple_values"
    assert err.info["argument"] == "title"


def test_extra_positional_arguments_allowed():
    env = make_env()
    src = (
        "{% macro f(n) contract {'required': ['n'], 'extra_positional': true} %}"
        "{{ ([n] + (varargs | list)) | sum }}{% endmacro %}"
    )
    assert env.from_string(src + "{{ f(10, 1, 2, 3) }}").render() == "16"


def test_extra_keyword_arguments_allowed():
    env = make_env()
    src = (
        "{% macro f(n) contract {'required': ['n'], 'extra_keyword': true} %}"
        "{{ n }}:{{ kwargs | sort | join(',') }}{% endmacro %}"
    )
    assert env.from_string(src + "{{ f(1, x=2, y=3) }}").render() == "1:x,y"


# ---------------------------------------------------------------------------
# caller
# ---------------------------------------------------------------------------


WRAP = (
    "{% macro wrap(title) "
    "contract {'required': ['title'], 'require_caller': true} %}"
    "<<{{ title }}:{{ caller() }}>>{% endmacro %}"
)


def test_caller_provided_by_call_block(import_env):
    t = import_env.from_string(
        "{% from 'm.html' import wrap %}{% call wrap('T') %}body{% endcall %}"
    )
    assert t.render() == "<<T:body>>"


def test_missing_caller():
    env = make_env()
    t = env.from_string(WRAP + "{{ wrap('T') }}")
    with pytest.raises(MacroContractError) as exc:
        t.render()
    assert exc.value.reason == "missing_caller"
    assert exc.value.macro_name == "wrap"


def test_unexpected_caller():
    env = make_env()
    src = (
        "{% macro f(a) contract {'required': ['a']} %}{{ a }}{% endmacro %}"
        "{{ f(1, caller=noop) }}"
    )
    with pytest.raises(MacroContractError) as exc:
        env.from_string(src).render(noop=lambda: "x")
    assert exc.value.reason == "unexpected_keyword_argument"
    assert exc.value.info["argument"] == "caller"


def test_explicit_caller_parameter_requirement():
    env = make_env()
    src = (
        "{% macro w(caller=none) contract {'require_caller': true} %}"
        "<{{ caller() }}>{% endmacro %}"
        "{% call w() %}hi{% endcall %}"
    )
    assert env.from_string(src).render() == "<hi>"


# ---------------------------------------------------------------------------
# imports / aliases / dynamic macros / extends / include
# ---------------------------------------------------------------------------


@pytest.fixture
def import_env():
    return Environment(
        loader=DictLoader(
            {
                "m.html": CARD + "\n" + WRAP,
                "base.html": "{% block content %}base{% endblock %}",
                "inc.html": (
                    "{% macro p(a) contract {'required': ['a']} %}P={{ a }}"
                    "{% endmacro %}{{ p(3) }}"
                ),
                "inc_bad.html": (
                    "{% macro p(a) contract {'required': ['a']} %}P={{ a }}"
                    "{% endmacro %}{{ p() }}"
                ),
            }
        ),
        macro_contracts=True,
    )


def test_from_import_enforces_contract(import_env):
    t = import_env.from_string("{% from 'm.html' import card %}{{ card('Z') }}")
    assert t.render() == "<Z|->"
    bad = import_env.from_string("{% from 'm.html' import card %}{{ card() }}")
    with pytest.raises(MacroContractError) as exc:
        bad.render()
    assert exc.value.macro_name == "card"


def test_namespace_alias_and_dynamic_macro(import_env):
    t = import_env.from_string(
        "{% import 'm.html' as m %}"
        "{% set a = m.card %}"
        "{% set d = m['card'] %}"
        "{{ a('C') }}|{{ d('D') }}"
    )
    assert t.render() == "<C|->|<D|->"
    bad = import_env.from_string(
        "{% import 'm.html' as m %}{% set d = m['card'] %}{{ d() }}"
    )
    with pytest.raises(MacroContractError) as exc:
        bad.render()
    assert exc.value.reason == "missing_argument"


def test_contract_survives_macro_passing_as_argument(import_env):
    t = import_env.from_string(
        "{% from 'm.html' import card %}"
        "{% macro invoke(fn, value) %}{{ fn(value) }}{% endmacro %}"
        "{{ invoke(card, 'Q') }}"
    )
    assert t.render() == "<Q|->"
    bad = import_env.from_string(
        "{% from 'm.html' import card %}"
        "{% macro invoke(fn) %}{{ fn() }}{% endmacro %}"
        "{{ invoke(card) }}"
    )
    with pytest.raises(MacroContractError) as exc:
        bad.render()
    assert exc.value.reason == "missing_argument"


def test_extends_with_contracted_macro(import_env):
    import_env.loader.mapping["child.html"] = (
        "{% extends 'base.html' %}"
        "{% macro f(a) contract {'required': ['a']} %}F={{ a }}{% endmacro %}"
        "{% block content %}{{ f(9) }}{% endblock %}"
    )
    assert import_env.get_template("child.html").render() == "F=9"


def test_include_contract(import_env):
    assert import_env.from_string("{% include 'inc.html' %}").render() == "P=3"
    with pytest.raises(MacroContractError) as exc:
        import_env.from_string("{% include 'inc_bad.html' %}").render()
    assert exc.value.reason == "missing_argument"


# ---------------------------------------------------------------------------
# async / sync requirement
# ---------------------------------------------------------------------------


def test_async_required_in_sync_env_fails():
    env = make_env()
    src = (
        "{% macro f(a) contract {'required': ['a'], 'awaitable': 'async'} %}"
        "{{ a }}{% endmacro %}{{ f(1) }}"
    )
    with pytest.raises(MacroContractError) as exc:
        env.from_string(src).render()
    assert exc.value.reason == "awaitable_mismatch"
    assert exc.value.info == {"expected": "async", "actual": "sync"}


def test_sync_required_in_async_env_fails(run_async_fn):
    env = Environment(enable_async=True, macro_contracts=True)
    src = (
        "{% macro f(a) contract {'required': ['a'], 'awaitable': 'sync'} %}"
        "{{ a }}{% endmacro %}{{ f(1) }}"
    )
    t = env.from_string(src)

    async def run():
        with pytest.raises(MacroContractError) as exc:
            await t.render_async()
        return exc.value

    err = run_async_fn(run)
    assert err.reason == "awaitable_mismatch"
    assert err.info == {"expected": "sync", "actual": "async"}


def test_async_required_in_async_env_works(run_async_fn):
    env = Environment(enable_async=True, macro_contracts=True)
    src = (
        "{% macro f(a) contract {'required': ['a'], 'awaitable': 'async'} %}"
        "{{ a }}{% endmacro %}{{ f(7) }}"
    )
    t = env.from_string(src)

    async def run():
        return await t.render_async()

    assert run_async_fn(run) == "7"


# ---------------------------------------------------------------------------
# non mutation / non consumption guarantees
# ---------------------------------------------------------------------------


def test_checker_does_not_consume_generator():
    env = make_env()
    macro = env.from_string(
        "{% macro f(a, b) contract {'required': ['a', 'b']} %}{{ a }}{{ b }}"
        "{% endmacro %}"
    ).module.f
    gen = (i for i in (7, 8))
    with pytest.raises(MacroContractError) as exc:
        macro(gen)  # missing b
    assert "generator" in exc.value.call["args"][0]
    # the generator must still be fully iterable, i.e. was not consumed
    assert list(gen) == [7, 8]


def test_checker_does_not_consume_caller():
    env = make_env()
    calls = []

    def caller():  # pragma: no cover - must never be called by the checker
        calls.append("called")
        return "noop"

    macro = env.from_string(
        "{% macro f(a) contract {'required': ['a']} %}{{ a }}{% endmacro %}"
    ).module.f
    with pytest.raises(MacroContractError):
        macro(1, caller=caller)
    assert calls == []


def test_checker_does_not_mutate_user_kwargs():
    env = make_env()
    macro = env.from_string(
        "{% macro f(a) contract {'required': ['a']} %}{{ a }}{% endmacro %}"
    ).module.f
    user_kwargs = {"a": 1, "bogus": 2}
    with pytest.raises(MacroContractError):
        macro(**user_kwargs)
    assert user_kwargs == {"a": 1, "bogus": 2}


def test_contract_never_mutates_macro_object():
    env = make_env()
    macro = env.from_string(
        "{% macro f(a, b='x') contract {'required': ['a'], 'optional': {'b': 'x'}}"
        " %}{{ a }}{{ b }}{% endmacro %}"
    ).module.f
    before = dict(vars(macro))
    assert not hasattr(macro, "_macro_contract")
    assert macro in env.macro_contracts
    macro("ok")
    with pytest.raises(MacroContractError):
        macro()
    after = dict(vars(macro))
    assert before == after
    assert list(macro.arguments) == ["a", "b"]
    assert macro.catch_kwargs is False
    assert macro.catch_varargs is False


# ---------------------------------------------------------------------------
# python api for dynamic macros
# ---------------------------------------------------------------------------


def test_python_bind_api():
    env = make_env()
    src = "{% macro h(x, y=1) %}{{ x }}/{{ y }}{% endmacro %}"
    macro = env.from_string(src).module.h
    contract = env.macro_contracts.bind(
        macro, {"required": ["x"], "optional": {"y": 1}}
    )
    assert contract.name == "h"
    assert contract.parameters == ("x", "y")
    assert macro(2) == "2/1"
    with pytest.raises(MacroContractError) as exc:
        macro()
    assert exc.value.reason == "missing_argument"


def test_bind_rejects_signature_mismatch():
    env = make_env()
    macro = env.from_string("{% macro h(x) %}{{ x }}{% endmacro %}").module.h
    with pytest.raises(TypeError):
        env.macro_contracts.bind(macro, {"required": ["x"], "optional": {"z": 1}})


# ---------------------------------------------------------------------------
# sandbox
# ---------------------------------------------------------------------------


def test_contracts_work_in_sandbox():
    env = SandboxedEnvironment(macro_contracts=True)
    ok = (
        "{% macro f(a) contract {'required': ['a']} %}[{{ a }}]{% endmacro %}{{ f(3) }}"
    )
    bad = (
        "{% macro f(a) contract {'required': ['a']} %}[{{ a }}]{% endmacro %}{{ f() }}"
    )
    assert env.from_string(ok).render() == "[3]"
    with pytest.raises(MacroContractError) as exc:
        env.from_string(bad).render()
    assert exc.value.reason == "missing_argument"


# ---------------------------------------------------------------------------
# compile time definition errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "{% macro f(a) contract {'required': ['a'], 'bogus': 1} %}{% endmacro %}",
        "{% macro f(a, b) contract {'required': ['a']} %}{% endmacro %}",
        "{% macro f(a) contract {'required': ['a', 'z']} %}{% endmacro %}",
        "{% macro f(a) contract"
        " {'required': ['a'], 'optional': {'a': 1}} %}{% endmacro %}",
        "{% macro f(a) contract {'required': x} %}{% endmacro %}",
        "{% macro f(a) contract {'required': ['a'], 'require_caller': true} %}"
        "{{ a }}{% endmacro %}",
        "{% macro f(a) contract {'required': ['a'], 'awaitable': 'later'}"
        " %}{% endmacro %}",
    ],
)
def test_invalid_contract_is_compile_error(source):
    env = make_env()
    with pytest.raises((TemplateAssertionError, TemplateSyntaxError)):
        env.from_string(source)


def test_contract_must_be_dict_literal():
    env = make_env()
    with pytest.raises(TemplateSyntaxError):
        env.from_string("{% macro f(a) contract [1, 2] %}{% endmacro %}")


# ---------------------------------------------------------------------------
# environment plumbing
# ---------------------------------------------------------------------------


def test_environment_pickles_with_registry():
    env = make_env()
    restored = pickle.loads(pickle.dumps(env))
    assert restored.macro_contract_enabled is True
    assert len(restored.macro_contracts) == 0
