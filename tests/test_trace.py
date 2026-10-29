import asyncio

import pytest

from jinja2 import DictLoader
from jinja2 import Environment
from jinja2 import Template
from jinja2.nativetypes import NativeEnvironment
from jinja2.sandbox import SandboxedEnvironment
from jinja2.trace import BLOCK_ENTER
from jinja2.trace import BLOCK_EXIT
from jinja2.trace import CANCELLED
from jinja2.trace import current_session
from jinja2.trace import EVENT_LIMIT
from jinja2.trace import EXTENDS
from jinja2.trace import IMPORT_ENTER
from jinja2.trace import INCLUDE_ENTER
from jinja2.trace import INCLUDE_EXIT
from jinja2.trace import MACRO_ENTER
from jinja2.trace import MACRO_EXIT
from jinja2.trace import OUTPUT
from jinja2.trace import OUTPUT_LIMIT
from jinja2.trace import RENDER_END
from jinja2.trace import RENDER_ERROR
from jinja2.trace import RENDER_START
from jinja2.trace import RenderTrace
from jinja2.trace import RenderTraceError
from jinja2.trace import TEMPLATE_ENTER
from jinja2.trace import TEMPLATE_ERROR
from jinja2.trace import TraceCancelledError
from jinja2.trace import TraceLimitError
from jinja2.trace import TraceObserver
from jinja2.trace import TraceRejectedError

TEMPLATES = {
    "base.html": (
        "B[{% block content %}base{% endblock %}]{% block footer %}F{% endblock %}"
    ),
    "inc_ctx.html": "inc({{ x }})",
    "inc_noctx.html": "incn",
    "mac.html": (
        "{% macro greet(name) %}HI {{ name | upper }}"
        "{% if name | length > 2 %}!{% endif %}"
        "{{ caller() }}{% endmacro %}"
        "{% macro plain() %}P{% endmacro %}"
    ),
    "child.html": (
        "{% extends 'base.html' %}"
        "{% block content %}"
        "Hi {{ name }}"
        "{% include 'inc_ctx.html' %}"
        "{% include 'inc_noctx.html' without context %}"
        "{% import 'mac.html' as mac with context %}"
        "{% from 'mac.html' import plain as p %}"
        "{% call() mac.greet(name) %}yo{% endcall %}{{ p() }}"
        "{% endblock %}"
    ),
}


@pytest.fixture
def env():
    return Environment(enable_trace=True, loader=DictLoader(TEMPLATES))


@pytest.fixture
def plain_env():
    return Environment(loader=DictLoader(TEMPLATES))


def kinds(trace, *wanted):
    return [(e.kind, e.target) for e in trace.events if e.kind in wanted or not wanted]


# ---------------------------------------------------------------------------
# default off
# ---------------------------------------------------------------------------


def test_disabled_by_default():
    env = Environment()
    assert env.enable_trace is False
    src = env.compile("{{ x }}", "t.html", raw=True)
    assert "_trace" not in src
    assert "__jinja_trace__" not in src


def test_enabled_compiles_instrumentation():
    env = Environment(enable_trace=True)
    src = env.compile("{{ x }}", "t.html", raw=True)
    assert "__jinja_trace__ = True" in src
    assert "_trace.template_enter" in src
    assert "_trace.template_exit" in src


def test_trace_requires_enabled_environment(plain_env):
    t = plain_env.from_string("{{ x }}")
    with pytest.raises(RuntimeError, match="enable_trace"):
        t.render(trace=RenderTrace())


def test_enabled_without_session_unchanged(env, plain_env):
    assert env.from_string("{{ x }}").render(x="a") == "a"
    assert env.from_string("{% include 'inc_ctx.html' %}").render(
        x="a"
    ) == plain_env.from_string("{% include 'inc_ctx.html' %}").render(x="a")


# ---------------------------------------------------------------------------
# basic event stream
# ---------------------------------------------------------------------------


def test_basic_render_events(env):
    t = env.from_string("a{{ x }}b")
    trace = RenderTrace()
    assert t.render(x="X", trace=trace) == "aXb"

    stream = trace.events
    assert stream[0].kind == RENDER_START
    assert stream[-1].kind == RENDER_END
    assert all(e.is_async is False for e in stream)
    assert [e.kind for e in stream if e.kind == OUTPUT] == [OUTPUT] * 3
    fragments = [e.fragment for e in stream if e.kind == OUTPUT]
    assert fragments == ["a", "X", "b"]
    assert trace.output_bytes == 3
    assert trace.output_count == 3
    # sequence numbers are contiguous from zero
    assert [e.seq for e in stream] == list(range(len(stream)))


def test_output_bytes_utf8(env):
    t = env.from_string("{{ x }}")
    trace = RenderTrace()
    t.render(x="é", trace=trace)
    out = next(e for e in trace.events if e.kind == OUTPUT)
    assert out.fragment == "é"
    assert out.length == 2
    assert trace.output_bytes == 2


def test_depth_and_template_identity(env):
    t = env.get_template("child.html")
    trace = RenderTrace()
    t.render(name="joe", x="X", trace=trace)

    # Every event has contiguous seq and a known template identity.
    for i, event in enumerate(trace.events):
        assert event.seq == i
        assert event.template is not None or event.kind == RENDER_START

    enter = [e for e in trace.events if e.kind == TEMPLATE_ENTER]
    names = {e.target for e in enter}
    # with-context include executes the template root; the without
    # context include reuses the cached module body, so it only has an
    # include boundary and no separate template enter.
    assert {"child.html", "base.html", "inc_ctx.html"} <= names
    assert "inc_noctx.html" not in names
    includes = {e.target for e in trace.events if e.kind == INCLUDE_ENTER}
    assert {"inc_ctx.html", "inc_noctx.html"} <= includes
    # nesting strictly increases and returns for the included template
    inc_enter = next(e for e in trace.events if e.kind == INCLUDE_ENTER)
    assert inc_enter.depth >= 2


def test_extends_events(env):
    trace = RenderTrace()
    env.get_template("child.html").render(name="x", trace=trace)
    extends = [e for e in trace.events if e.kind == EXTENDS]
    assert len(extends) == 1
    assert extends[0].target == "base.html"


def test_include_with_and_without_context(env):
    trace = RenderTrace()
    out = env.get_template("child.html").render(name="jo", x="X", trace=trace)
    assert "inc(X)" in out
    assert "incn" in out
    includes = [e for e in trace.events if e.kind == INCLUDE_ENTER]
    assert {e.target for e in includes} == {"inc_ctx.html", "inc_noctx.html"}
    by_name = {e.target: e for e in includes}
    assert by_name["inc_ctx.html"].with_context is True
    assert by_name["inc_noctx.html"].with_context is False
    # every enter has a matching exit in order
    assert kinds(trace, INCLUDE_ENTER, INCLUDE_EXIT) == [
        (INCLUDE_ENTER, "inc_ctx.html"),
        (INCLUDE_EXIT, "inc_ctx.html"),
        (INCLUDE_ENTER, "inc_noctx.html"),
        (INCLUDE_EXIT, "inc_noctx.html"),
    ]


def test_import_events_and_caching(env):
    t = env.from_string(
        "{% import 'mac.html' as a %}{% import 'mac.html' as b %}"
        "{% from 'mac.html' import plain as c %}"
        "{{ a.plain() }}{{ b.plain() }}{{ c() }}"
    )
    trace = RenderTrace()
    assert t.render(trace=trace) == "PPP"
    imports = [e for e in trace.events if e.kind == IMPORT_ENTER]
    assert [e.target for e in imports] == ["mac.html"] * 3
    # first import renders the module, later ones reuse it
    assert imports[0].cached is False
    assert imports[1].cached is True
    assert imports[2].cached is True
    assert imports[0].with_context is False


def test_import_with_context_is_not_cached(env):
    t = env.from_string("{% import 'mac.html' as a with context %}{{ a.plain() }}")
    trace = RenderTrace()
    t.render(trace=trace)
    imp = next(e for e in trace.events if e.kind == IMPORT_ENTER)
    assert imp.cached is False
    assert imp.with_context is True


def test_block_boundaries_and_super(env):
    t = env.from_string(
        "{% extends 'base.html' %}{% block content %}"
        "{{ super() }}->override{% endblock %}"
    )
    trace = RenderTrace()
    assert t.render(trace=trace) == "B[base->override]F"
    blocks = [e.target for e in trace.events if e.kind == BLOCK_ENTER]
    assert "content" in blocks and "footer" in blocks
    pairs = kinds(trace, BLOCK_ENTER, BLOCK_EXIT)
    # every enter is followed by its exit
    stack = []
    for kind, name in pairs:
        if kind == BLOCK_ENTER:
            stack.append(name)
        else:
            assert stack.pop() == name
    assert not stack


def test_macro_calls_caller_and_aliases_share_session(env):
    t = env.from_string(
        "{% import 'mac.html' as m %}"
        "{% from 'mac.html' import greet as g, plain as p %}"
        "{% call() m.greet('tom') %}<{{ p() }}>{% endcall %}"
        "{% call() g('ben') %}!{% endcall %}"
    )
    trace = RenderTrace()
    out = t.render(trace=trace)
    assert out == "HI TOM!<P>HI BEN!!"
    macros = [e for e in trace.events if e.kind == MACRO_ENTER]
    # greet twice (one through m, one through alias g), plain twice,
    # plus two anonymous callers
    assert [(e.target, e.origin) for e in macros] == [
        ("greet", "mac.html"),
        (None, None),
        ("plain", "mac.html"),
        ("greet", "mac.html"),
        (None, None),
    ]
    # all macros balance
    assert sum(1 for e in trace.events if e.kind == MACRO_ENTER) == sum(
        1 for e in trace.events if e.kind == MACRO_EXIT
    )


def test_events_in_execution_order(env):
    t = env.from_string("1{% include 'inc_noctx.html' %}2")
    trace = RenderTrace()
    t.render(trace=trace)
    seq = [(e.kind, e.target or e.fragment) for e in trace.events]
    assert seq[0][0] == RENDER_START
    assert seq[1][0] == TEMPLATE_ENTER
    assert (OUTPUT, "1") in seq
    ordered = [k for k, _ in seq]
    i_enter = ordered.index(INCLUDE_ENTER)
    i_exit = ordered.index(INCLUDE_EXIT)
    # output produced by the include sits between enter and exit
    inner = [e for e in trace.events if i_enter < e.seq < i_exit]
    assert any(e.kind == OUTPUT and e.fragment == "incn" for e in inner)


def test_dynamic_template_object(env):
    dyn = env.from_string("D-{{ y }}")
    t = env.from_string("a{% include obj %}b")
    trace = RenderTrace()
    assert t.render(obj=dyn, y=7, trace=trace) == "aD-7b"
    # the dynamic memory template gets its own identity
    dyn_events = [
        e for e in trace.events if e.template and e.template.startswith("memory:")
    ]
    assert dyn_events
    assert any(e.kind == INCLUDE_ENTER for e in trace.events)


def test_dynamic_extends(env):
    t = env.from_string("{% extends parent %}{% block content %}C{% endblock %}")
    trace = RenderTrace()
    assert t.render(parent="base.html", trace=trace) == "B[C]F"
    assert any(e.kind == EXTENDS and e.target == "base.html" for e in trace.events)


def test_ignore_missing_no_events(env):
    t = env.from_string("a{% include 'nope.html' ignore missing %}b")
    trace = RenderTrace()
    assert t.render(trace=trace) == "ab"
    assert not [e for e in trace.events if e.kind == INCLUDE_ENTER]


# ---------------------------------------------------------------------------
# one session per render, no reuse
# ---------------------------------------------------------------------------


def test_repeated_renders_get_independent_sessions(env):
    t = env.from_string("{% import 'mac.html' as m %}{{ m.plain() }}")
    tr1 = RenderTrace()
    tr2 = RenderTrace()
    assert t.render(trace=tr1) == "P"
    assert t.render(trace=tr2) == "P"
    assert len(tr1.events) == len(tr2.events)
    assert tr1.events[0].seq == tr2.events[0].seq == 0
    assert tr1 is not tr2
    # identity of seqs is independent
    assert {e.seq for e in tr2.events} == {e.seq for e in tr1.events}


def test_session_cannot_be_reused(env):
    t = env.from_string("{{ 1 }}")
    trace = RenderTrace()
    t.render(trace=trace)
    with pytest.raises(RuntimeError, match="one render"):
        t.render(trace=trace)


def test_callable_observer(env):
    seen = []
    t = env.from_string("{{ 1 }}")
    trace = RenderTrace(observer=seen.append)
    t.render(trace=trace)
    assert seen and seen[0].kind == RENDER_START
    # callable observer does not store events itself
    assert isinstance(seen[0], type(trace.events[0]))


def test_observer_class(env):
    class Rec(TraceObserver):
        def __init__(self):
            self.items = []

        def on_event(self, event):
            self.items.append(event)

    rec = Rec()
    t = env.from_string("x")
    t.render(trace=RenderTrace(observer=rec, record=False))
    assert [e.kind for e in rec.items][:2] == [RENDER_START, TEMPLATE_ENTER]


# ---------------------------------------------------------------------------
# limits, rejection, cancellation, errors
# ---------------------------------------------------------------------------


def test_event_limit(env):
    t = env.from_string("{% for i in range(10) %}{{ i }}{% endfor %}")
    trace = RenderTrace(max_events=4)
    with pytest.raises(TraceLimitError) as exc:
        t.render(trace=trace)
    err = exc.value
    assert err.reason == EVENT_LIMIT
    assert err.limit == 4
    assert isinstance(err.seq, int)
    assert err.template is not None
    error_event = trace.events[-1]
    assert error_event.kind == RENDER_ERROR
    assert error_event.error["reason"] == EVENT_LIMIT


def test_output_byte_limit():
    env = Environment(enable_trace=True)
    t = env.from_string("{% for i in range(10) %}x{% endfor %}")
    trace = RenderTrace(max_output_bytes=4)
    with pytest.raises(TraceLimitError) as exc:
        t.render(trace=trace)
    err = exc.value
    assert err.reason == OUTPUT_LIMIT
    assert err.output_bytes == 4
    assert err.output_count == 4


def test_observer_rejects_false(env):
    trace = RenderTrace(observer=lambda e: e.kind != INCLUDE_ENTER)
    t = env.from_string("a{% include 'inc_noctx.html' %}b")
    with pytest.raises(TraceRejectedError):
        t.render(trace=trace)


def test_observer_raises_is_rejection(env):
    def obs(event):
        if event.kind == OUTPUT:
            raise ValueError("boom")

    trace = RenderTrace(observer=obs)
    t = env.from_string("{{ 1 }}")
    with pytest.raises(TraceRejectedError) as exc:
        t.render(trace=trace)
    assert isinstance(exc.value.original, ValueError)


def test_cancel_session(env):
    t = env.from_string("{% for i in range(10) %}{{ i }}{% endfor %}")
    holder: dict = {}

    def obs(event):
        if event.seq >= 4:
            holder["s"].cancel()
        return True

    trace = RenderTrace(observer=obs)
    holder["s"] = trace
    with pytest.raises(TraceCancelledError) as exc:
        t.render(trace=trace)
    assert exc.value.reason == CANCELLED


def test_cancel_callback():
    env = Environment(enable_trace=True)
    t = env.from_string("{% for i in range(10) %}{{ i }}{% endfor %}")
    count = 0

    def cancelled():
        nonlocal count
        count += 1
        return count > 3

    trace = RenderTrace(is_cancelled=cancelled)
    with pytest.raises(TraceCancelledError):
        t.render(trace=trace)


def test_template_error_is_structured(env):
    t = env.from_string("a{{ 1 / 0 }}b")
    trace = RenderTrace()
    with pytest.raises(RenderTraceError) as exc:
        t.render(trace=trace)
    err = exc.value
    assert err.reason == TEMPLATE_ERROR
    assert isinstance(err.original, ZeroDivisionError)
    assert err.__cause__ is err.original
    assert err.output_count == 1  # "a" was produced before the failure
    assert err.output_bytes == 1
    assert err.template is not None
    assert trace.events[-1].kind == RENDER_ERROR


def test_error_does_not_roll_back_stream():
    env = Environment(enable_trace=True)
    t = env.from_string(
        "{% for i in range(10) %}{{ i }}"
        "{% if i == 4 %}{{ 1 / 0 }}{% endif %}{% endfor %}"
    )
    trace = RenderTrace()
    received = []
    with pytest.raises(RenderTraceError):
        for piece in t.generate(trace=trace):
            received.append(piece)
    # pieces yielded before the failure stay with the caller
    assert received == ["0", "1", "2", "3", "4"]


def test_stream_byte_limit_keeps_delivered_pieces():
    env = Environment(enable_trace=True)
    t = env.from_string("{% for i in range(10) %}[{{ i }}]{% endfor %}")
    trace = RenderTrace(max_output_bytes=6)
    received = []
    stream = t.stream(trace=trace)
    with pytest.raises(TraceLimitError):
        for piece in stream:
            received.append(piece)
    assert "".join(received) == "[0][1]"


def test_stream_abandon_marked_closed():
    env = Environment(enable_trace=True)
    t = env.from_string("{% for i in range(10) %}{{ i }}{% endfor %}")
    trace = RenderTrace()
    gen = t.generate(trace=trace)
    assert next(gen) == "0"
    gen.close()
    last = trace.events[-1]
    assert last.kind == RENDER_ERROR
    assert last.error["reason"] == "closed"


def test_generate_entry_label(env):
    t = env.from_string("{{ 1 }}")
    trace = RenderTrace()
    list(t.generate(trace=trace))
    assert trace.entry == "generate"
    trace = RenderTrace()
    list(t.stream(trace=trace))
    assert trace.entry == "stream"


def test_template_stream_buffering(env):
    t = env.from_string("{% for i in range(6) %}{{ i }}{% endfor %}")
    trace = RenderTrace()
    stream = t.stream(trace=trace)
    stream.enable_buffering(3)
    out = list(stream)
    assert "".join(out) == "012345"
    assert trace.output_count == 6


# ---------------------------------------------------------------------------
# tracing does not duplicate work or mutate objects
# ---------------------------------------------------------------------------


def test_no_duplicate_execution():
    loader = DictLoader(
        {
            "m.html": "{% macro body() %}[{{ caller() }}]{% endmacro %}",
        }
    )

    def build(enable_trace):
        counts = {"filter": 0, "test": 0, "func": 0}
        env = Environment(enable_trace=enable_trace, loader=loader)

        def myfilter(value):
            counts["filter"] += 1
            return value.upper()

        def mytest(value):
            counts["test"] += 1
            return value

        def myfunc():
            counts["func"] += 1
            return "uf"

        env.filters["up"] = myfilter
        env.tests["truth"] = mytest
        env.globals["uf"] = myfunc

        src = (
            "{% import 'm.html' as m %}"
            "{% call() m.body() %}"
            "{{ uf() }}{{ 'a' | up }}{% if 's' is truth %}t{% endif %}"
            "{% endcall %}"
        )
        return env, counts, src

    env0, counts0, src = build(False)
    base = env0.from_string(src).render()

    env1, counts1, src = build(True)
    t = env1.from_string(src)
    out_traced = t.render(trace=RenderTrace())
    traced_calls = dict(counts1)
    out_untraced = t.render()

    assert out_traced == base == "[ufAt]"
    assert out_untraced == base
    assert traced_calls == counts0 == {"filter": 1, "test": 1, "func": 1}


def test_macro_body_executes_once_per_call(env):
    t = env.from_string(
        "{% macro m() %}{% set x = bump() %}{{ x }}{% endmacro %}{{ m() }}{{ m() }}"
    )
    counter = {"n": 0}

    def bump():
        counter["n"] += 1
        return counter["n"]

    trace = RenderTrace()
    out = t.render(bump=bump, trace=trace)
    assert out == "12"
    assert counter["n"] == 2
    assert len([e for e in trace.events if e.kind == MACRO_ENTER]) == 2


def test_include_inside_cached_module_macro():
    # A macro defined in a cached (``without context``) imported module
    # closes over a detached context; its internal include must still be
    # traced by the session driving the render, on every render.
    loader = DictLoader(
        {
            "inc.html": "I",
            "lib.html": ("{% macro m() %}L{% include 'inc.html' %}X{% endmacro %}"),
        }
    )

    def render():
        env = Environment(enable_trace=True, loader=loader)
        t = env.from_string("{% import 'lib.html' as lib %}a{{ lib.m() }}b")
        trace = RenderTrace()
        out = t.render(trace=trace)
        kinds = [(e.kind, e.target) for e in trace.events]
        assert (MACRO_ENTER, "m") in kinds
        enter = kinds.index((MACRO_ENTER, "m"))
        exit_ = kinds.index((MACRO_EXIT, "m"))
        inner = kinds[enter:exit_]
        assert (INCLUDE_ENTER, "inc.html") in inner
        assert (INCLUDE_EXIT, "inc.html") in inner
        assert (TEMPLATE_ENTER, "inc.html") in inner
        return out

    assert render() == "aLIXb"
    assert render() == "aLIXb"


def test_context_and_template_not_mutated(env):
    t = env.get_template("child.html")
    before = set(t.blocks)
    trace = RenderTrace()
    t.render(name="x", trace=trace)
    assert set(t.blocks) == before
    # rendered output never ends up stored on the session beyond events
    assert trace.output_bytes == len(
        "".join(e.fragment or "" for e in trace.events if e.kind == OUTPUT)
    )


def test_trace_error_state_contents(env):
    t = env.from_string("ab{{ 1 / 0 }}")
    trace = RenderTrace()
    with pytest.raises(RenderTraceError) as exc:
        t.render(trace=trace)
    state = exc.value.state
    assert state["reason"] == TEMPLATE_ERROR
    assert {"seq", "template", "output_bytes", "output_count", "event_count"} <= set(
        state
    )


# ---------------------------------------------------------------------------
# other environments
# ---------------------------------------------------------------------------


def test_native_environment():
    env = NativeEnvironment(enable_trace=True)
    trace = RenderTrace()
    result = env.from_string("{{ 40 + 2 }}").render(trace=trace)
    assert result == 42
    assert trace.output_count == 1


def test_sandboxed_environment():
    env = SandboxedEnvironment(enable_trace=True)
    trace = RenderTrace()
    assert env.from_string("{{ 1 + 2 }}").render(trace=trace) == "3"
    assert any(e.kind == OUTPUT for e in trace.events)


# ---------------------------------------------------------------------------
# async
# ---------------------------------------------------------------------------


def test_async_render(run_async_fn):
    env = Environment(
        enable_trace=True, enable_async=True, loader=DictLoader(TEMPLATES)
    )

    async def run():
        trace = RenderTrace()
        t = env.get_template("child.html")
        out = await t.render_async(name="jo", trace=trace)
        return out, trace

    out, trace = run_async_fn(run)
    assert "HI JO" in out
    assert trace.events[0].kind == RENDER_START
    assert all(e.is_async for e in trace.events)
    assert trace.events[-1].kind == RENDER_END


def test_async_sync_wrapper_uses_same_session(run_async_fn):
    env = Environment(enable_trace=True, enable_async=True)

    async def run():
        trace = RenderTrace()
        out = await env.from_string(
            "{{ 1 }}{% macro x() %}m{% endmacro %}{{ x() }}"
        ).render_async(trace=trace)
        return out, trace

    out, trace = run_async_fn(run)
    assert out == "1m"
    assert all(e.is_async for e in trace.events)


def test_async_generate(run_async_fn):
    env = Environment(enable_trace=True, enable_async=True)

    async def run():
        t = env.from_string("{% for i in range(4) %}{{ i }}{% endfor %}")
        trace = RenderTrace()
        pieces = []
        async for piece in t.generate_async(trace=trace):
            pieces.append(piece)
        return pieces, trace

    pieces, trace = run_async_fn(run)
    assert pieces == ["0", "1", "2", "3"]
    assert trace.entry == "generate"


def test_async_task_cancelled(run_async_fn):
    env = Environment(enable_trace=True, enable_async=True)

    async def sleep():
        await asyncio.sleep(0.001)
        return "x"

    async def run():
        t = env.from_string("{% for i in range(1000) %}{{ sleep() }}{% endfor %}")
        trace = RenderTrace()
        task = asyncio.ensure_future(t.render_async(sleep=sleep, trace=trace))
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except TraceCancelledError as err:
            return err
        return None

    # task cancellation only applies under asyncio
    if run_async_fn.__name__ != "_asyncio_run":
        pytest.skip("task cancellation is asyncio specific")
    err = run_async_fn(run)
    assert isinstance(err, TraceCancelledError)


def test_async_template_error(run_async_fn):
    env = Environment(enable_trace=True, enable_async=True)

    async def run():
        t = env.from_string("a{{ 1 / 0 }}")
        trace = RenderTrace()
        try:
            await t.render_async(trace=trace)
        except RenderTraceError as err:
            return err

    err = run_async_fn(run)
    assert err.reason == TEMPLATE_ERROR
    assert isinstance(err.original, ZeroDivisionError)


# ---------------------------------------------------------------------------
# from_string memory identity stability
# ---------------------------------------------------------------------------


def test_memory_template_identity_stable():
    env = Environment(enable_trace=True)
    t = env.from_string("{{ 1 }}")
    trace = RenderTrace()
    t.render(trace=trace)
    root = trace.events[0].target
    assert root and root.startswith("memory:")
    assert all(e.template == root for e in trace.events if e.kind != RENDER_START)


def test_spontaneous_template_enable_trace():
    t = Template("{{ x }}", enable_trace=True)
    trace = RenderTrace()
    assert t.render(x=1, trace=trace) == "1"
    assert trace.events[0].kind == RENDER_START


def test_nested_untraced_render_is_isolated():
    env = Environment(enable_trace=True)
    inner = env.from_string("{% macro m() %}IM{% endmacro %}{{ m() }}")
    outer = env.from_string("O[{{ embed() }}]")

    def embed():
        # an untraced render executed inside an outer traced render
        assert current_session() is not None
        return inner.render()

    trace = RenderTrace()
    assert outer.render(embed=embed, trace=trace) == "O[IM]"
    # internals of the untraced nested render are not visible
    assert not [e for e in trace.events if e.kind == MACRO_ENTER]
    # but the string it produced is an output fragment of the outer render
    assert [e.fragment for e in trace.events if e.kind == OUTPUT] == [
        "O[",
        "IM",
        "]",
    ]


def test_async_include_without_context(run_async_fn):
    env = Environment(
        enable_trace=True,
        enable_async=True,
        loader=DictLoader({"inc.html": "INC"}),
    )
    t = env.from_string("a{% include 'inc.html' without context %}b")

    async def run():
        trace = RenderTrace()
        out = await t.render_async(trace=trace)
        return out, trace

    out, trace = run_async_fn(run)
    assert out == "aINCb"
    assert any(e.kind == INCLUDE_ENTER and e.target == "inc.html" for e in trace.events)
    assert all(e.is_async for e in trace.events)
