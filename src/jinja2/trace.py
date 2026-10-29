"""Render event tracing for :mod:`jinja2`.

This module implements an *opt-in* tracing facility that records the
complete execution of a single render call.  Tracing is switched on per
:class:`~jinja2.Environment` with ``enable_trace=True``; even on a
tracing enabled environment nothing is recorded unless a
:class:`RenderTrace` session is passed to the render entry point.

A session observes, in execution order:

* render lifecycle (``render_start`` / ``render_end`` / ``render_error``)
* template identities (``template_enter`` / ``template_exit``)
* inheritance (``extends``)
* includes (``include_enter`` / ``include_exit``)
* imports (``import_enter`` / ``import_exit``)
* macro calls (``macro_enter`` / ``macro_exit``)
* block boundaries (``block_enter`` / ``block_exit``)
* output fragments (``output``)

Every event carries a monotonic sequence number, the current template
identity, the nesting depth and whether the render executes
synchronously or asynchronously.

Tracing never re-executes filters, tests, macros, callers or user
functions, never mutates :class:`~jinja2.Template`,
:class:`~jinja2.runtime.Context`, the AST or user objects, and each
render gets a fresh, independent session.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
import typing as t

from markupsafe import Markup

if t.TYPE_CHECKING:
    from .environment import Template

#: Session currently driving a render in this execution context.
_current_session: contextvars.ContextVar[RenderTrace | None] = contextvars.ContextVar(
    "jinja2_render_trace", default=None
)


def current_session() -> RenderTrace | None:
    """Return the session driving the current render, if any."""
    return _current_session.get()


@contextlib.contextmanager
def no_session() -> t.Iterator[None]:
    """Hide any outer render session for nested, untraced renders."""
    token = _current_session.set(None)
    try:
        yield
    finally:
        _current_session.reset(token)


# ---------------------------------------------------------------------------
# Event kinds
# ---------------------------------------------------------------------------

RENDER_START = "render_start"
RENDER_END = "render_end"
RENDER_ERROR = "render_error"
TEMPLATE_ENTER = "template_enter"
TEMPLATE_EXIT = "template_exit"
EXTENDS = "extends"
INCLUDE_ENTER = "include_enter"
INCLUDE_EXIT = "include_exit"
IMPORT_ENTER = "import_enter"
IMPORT_EXIT = "import_exit"
MACRO_ENTER = "macro_enter"
MACRO_EXIT = "macro_exit"
BLOCK_ENTER = "block_enter"
BLOCK_EXIT = "block_exit"
OUTPUT = "output"

#: All known event kinds.
EVENT_KINDS: frozenset[str] = frozenset(
    {
        RENDER_START,
        RENDER_END,
        RENDER_ERROR,
        TEMPLATE_ENTER,
        TEMPLATE_EXIT,
        EXTENDS,
        INCLUDE_ENTER,
        INCLUDE_EXIT,
        IMPORT_ENTER,
        IMPORT_EXIT,
        MACRO_ENTER,
        MACRO_EXIT,
        BLOCK_ENTER,
        BLOCK_EXIT,
        OUTPUT,
    }
)

# ---------------------------------------------------------------------------
# Abort reasons
# ---------------------------------------------------------------------------

REJECTED = "rejected"
EVENT_LIMIT = "event_limit"
OUTPUT_LIMIT = "output_limit"
CANCELLED = "cancelled"
CLOSED = "closed"
TEMPLATE_ERROR = "template_error"


class TraceEvent:
    """A single event observed while rendering a template.

    Not every attribute is meaningful for every kind; unused attributes
    are ``None``.  Events are immutable value objects.
    """

    __slots__ = (
        "seq",
        "kind",
        "time",
        "template",
        "target",
        "depth",
        "is_async",
        "purpose",
        "fragment",
        "length",
        "output_bytes",
        "with_context",
        "cached",
        "names",
        "binding",
        "origin",
        "error",
    )

    def __init__(
        self,
        seq: int,
        kind: str,
        time: float,
        template: str | None,
        target: str | None,
        depth: int,
        is_async: bool,
        purpose: str,
        fragment: str | None = None,
        length: int | None = None,
        output_bytes: int | None = None,
        with_context: bool | None = None,
        cached: bool | None = None,
        names: tuple[str, ...] | None = None,
        binding: str | None = None,
        origin: str | None = None,
        error: dict[str, t.Any] | None = None,
    ) -> None:
        #: Zero based, per session sequence number.
        self.seq = seq
        #: One of the module level event kind constants.
        self.kind = kind
        #: Value of :func:`time.monotonic` when the event was emitted.
        self.time = time
        #: Identity of the template currently executing.
        self.template = template
        #: The entity the event refers to (another template, block or
        #: macro name depending on the kind).
        self.target = target
        #: Nesting depth, including templates, blocks and macro calls.
        self.depth = depth
        #: ``True`` when the render executes asynchronously.
        self.is_async = is_async
        #: ``"render"`` for output producing renders, ``"module"`` when a
        #: template is rendered to build an imported module.
        self.purpose = purpose
        #: The rendered fragment for :data:`OUTPUT` events.
        self.fragment = fragment
        #: UTF-8 byte length of ``fragment``.
        self.length = length
        #: Total response bytes emitted so far after the event.
        self.output_bytes = output_bytes
        self.with_context = with_context
        self.cached = cached
        #: Names requested by ``from ... import ...``.
        self.names = names
        #: Variable an imported template is bound to.
        self.binding = binding
        #: Template that defines a macro.
        self.origin = origin
        #: Error details for :data:`RENDER_ERROR` events.
        self.error = error

    @property
    def name(self) -> str | None:
        """Alias of :attr:`target`."""
        return self.target

    def to_dict(self) -> dict[str, t.Any]:
        """Return a plain ``dict`` representation of the event."""
        return {
            "seq": self.seq,
            "kind": self.kind,
            "time": self.time,
            "template": self.template,
            "target": self.target,
            "depth": self.depth,
            "is_async": self.is_async,
            "purpose": self.purpose,
            "fragment": self.fragment,
            "length": self.length,
            "output_bytes": self.output_bytes,
            "with_context": self.with_context,
            "cached": self.cached,
            "names": self.names,
            "binding": self.binding,
            "origin": self.origin,
            "error": self.error,
        }

    def __repr__(self) -> str:
        return f"<TraceEvent {self.seq} {self.kind!r} template={self.template!r}>"


class Frame:
    __slots__ = ("kind", "template", "name")

    def __init__(self, kind: str, template: str | None, name: str | None) -> None:
        self.kind = kind
        self.template = template
        self.name = name


class RenderTraceError(Exception):
    """Structured error raised when a traced render cannot finish.

    The error always reports the event sequence number at which it
    occurred, the template currently executing and how much output had
    already been produced.  Errors raised by the template itself are
    chained via :attr:`__cause__` and available as :attr:`original`.
    """

    reason = "trace_error"

    def __init__(
        self,
        message: str,
        *,
        seq: int,
        template: str | None,
        output_bytes: int,
        output_count: int,
        event_count: int,
        limit: int | None = None,
        original: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        #: Sequence number of the associated ``render_error`` event.
        self.seq = seq
        #: Identity of the template executing when the error occurred.
        self.template = template
        #: Number of UTF-8 bytes already handed to the caller.
        self.output_bytes = output_bytes
        #: Number of output fragments already handed to the caller.
        self.output_count = output_count
        #: Number of events emitted before the error.
        self.event_count = event_count
        #: The configured limit, for limit errors.
        self.limit = limit
        #: The original exception for template failures.
        self.original = original

    @property
    def state(self) -> dict[str, t.Any]:
        """Snapshot of the render state at the point of failure."""
        return {
            "reason": self.reason,
            "seq": self.seq,
            "template": self.template,
            "output_bytes": self.output_bytes,
            "output_count": self.output_count,
            "event_count": self.event_count,
            "limit": self.limit,
        }


class TraceRejectedError(RenderTraceError):
    """Raised when the event observer rejects an event."""

    reason = REJECTED


class TraceLimitError(RenderTraceError):
    """Raised when the event count or output byte limit is exceeded."""

    reason = EVENT_LIMIT


class TraceCancelledError(RenderTraceError):
    """Raised when a traced render is cancelled."""

    reason = CANCELLED


class TraceObserver:
    """Base class for render event observers.

    Subclass and override :meth:`on_event`.  Returning ``False`` or
    raising an exception from the callback aborts the render with a
    :class:`TraceRejectedError`.
    """

    def on_event(self, event: TraceEvent) -> bool | None:
        """Called for every emitted event.

        Return ``False`` (or raise) to reject the event and abort the
        render.  Any other return value continues the render.
        """
        return True


class _CallableObserver(TraceObserver):
    def __init__(self, func: t.Callable[[TraceEvent], t.Any]) -> None:
        self._func = func

    def on_event(self, event: TraceEvent) -> bool | None:
        rv = self._func(event)
        return True if rv is None else t.cast(bool, rv)


def template_identity(template: Template | None, name: str | None = None) -> str | None:
    """Return a stable identity for a template.

    Loaded templates use their load name; templates created from
    strings use a ``memory:`` identity derived from the object id.
    """
    if name is not None:
        return name
    if template is not None:
        template_name = t.cast("str | None", getattr(template, "name", None))
        if template_name is not None:
            return template_name
        return f"memory:{id(template):x}"
    return None


def normalize_trace(trace: t.Any) -> RenderTrace:
    """Coerce a user supplied ``trace`` argument into a session.

    Accepts an existing :class:`RenderTrace`, a :class:`TraceObserver`
    or a plain callable.
    """
    if isinstance(trace, RenderTrace):
        return trace
    if isinstance(trace, TraceObserver):
        return RenderTrace(observer=trace, record=False)
    if callable(trace):
        return RenderTrace(observer=_CallableObserver(trace), record=False)
    raise TypeError(
        "trace must be a RenderTrace, a TraceObserver or a callable,"
        f" got {type(trace).__name__!r}"
    )


class RenderTrace:
    """An independent tracing session for one render execution.

    A session may only be used once.  Create a new instance for every
    ``render``, ``render_async``, ``generate`` /
    :class:`~jinja2.TemplateStream` execution.

    :param observer: Optional :class:`TraceObserver` or callable notified
        for every event.
    :param max_events: Maximum number of regular events to record.
    :param max_output_bytes: Maximum number of UTF-8 output bytes.
    :param is_cancelled: Optional callable polled before every event; a
        true return value cancels the render.
    :param record: Keep emitted events in :attr:`events`. Defaults to
        ``True``.
    """

    #: ``True`` for real sessions, ``False`` for the shared null session.
    active = True

    def __init__(
        self,
        observer: TraceObserver | t.Callable[[TraceEvent], t.Any] | None = None,
        *,
        max_events: int | None = None,
        max_output_bytes: int | None = None,
        is_cancelled: t.Callable[[], bool] | None = None,
        record: bool = True,
    ) -> None:
        if max_events is not None and max_events < 0:
            raise ValueError("max_events must not be negative")
        if max_output_bytes is not None and max_output_bytes < 0:
            raise ValueError("max_output_bytes must not be negative")
        if isinstance(observer, TraceObserver):
            self._observer: TraceObserver | None = observer
        elif callable(observer):
            self._observer = _CallableObserver(observer)
        else:
            self._observer = None
        self.max_events = max_events
        self.max_output_bytes = max_output_bytes
        self._cancel_check = is_cancelled
        self._record = record

        self._bound = False
        self._finished = False
        self._failed = False
        self._cancel_requested = False
        self._error: RenderTraceError | None = None

        self._seq = 0
        self._frames: list[Frame] = []
        self._purpose_stack: list[str] = []
        self._events: list[TraceEvent] = []
        self._output_bytes = 0
        self._output_count = 0
        self._root_template: str | None = None
        self.entry: str | None = None
        self.is_async: bool = False
        self.session_id = id(self)

    # -- introspection -----------------------------------------------------

    @property
    def events(self) -> list[TraceEvent]:
        """All recorded events (empty unless ``record=True``)."""
        return list(self._events)

    @property
    def event_count(self) -> int:
        return self._seq

    @property
    def output_bytes(self) -> int:
        return self._output_bytes

    @property
    def output_count(self) -> int:
        return self._output_count

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def finished(self) -> bool:
        return self._finished or self._failed

    @property
    def depth(self) -> int:
        return len(self._frames)

    @property
    def current_template(self) -> str | None:
        return self._current_template()

    @property
    def state(self) -> dict[str, t.Any]:
        return {
            "seq": self._seq,
            "template": self._current_template(),
            "output_bytes": self._output_bytes,
            "output_count": self._output_count,
            "event_count": self._seq,
            "depth": len(self._frames),
        }

    def __iter__(self) -> t.Iterator[TraceEvent]:
        return iter(list(self._events))

    # -- cancellation ------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation.  The next event or output aborts."""
        self._cancel_requested = True

    @contextlib.contextmanager
    def activate(self) -> t.Iterator[RenderTrace]:
        """Make this session the current render session.

        Used by the render entry points so macro calls that reach the
        engine through a cached imported module (whose own context is
        detached from the current render) are still traced by the same
        session.
        """
        token = _current_session.set(self)
        try:
            yield self
        finally:
            _current_session.reset(token)

    # -- lifecycle ---------------------------------------------------------

    def begin(self, entry: str, template: Template, is_async: bool) -> None:
        if self._bound:
            raise RuntimeError(
                "A RenderTrace session can only be used for one render;"
                " create a new session for every render call"
            )
        self._bound = True
        self.entry = entry
        self.is_async = is_async
        self._root_template = template_identity(template)
        self._frames.append(Frame("root", self._root_template, None))
        self._emit(RENDER_START, target=self._root_template)

    def finish(self) -> None:
        """Emit the terminal ``render_end`` event.

        Never raises: observer failures or limits at this point do not
        invalidate an already completed render.
        """
        if not self._bound or self._finished or self._failed:
            return
        self._finished = True
        self._force_emit(RENDER_END, target=self._root_template)

    def mark_closed(self) -> None:
        """Record that a stream was abandoned before completion."""
        if self._finished or self._failed:
            return
        self._finished = True
        self._force_emit(
            RENDER_ERROR,
            target=self._current_template(),
            error={"reason": CLOSED, "type": "GeneratorExit"},
        )

    def fail(self, exc: BaseException) -> t.NoReturn:
        """Turn an exception into a structured :class:`RenderTraceError`."""
        if isinstance(exc, RenderTraceError):
            # Already structured (propagating through another wrapper).
            self._failed = True
            self._error = exc
            raise exc
        if self._failed and self._error is not None:
            raise self._error from exc
        if isinstance(exc, asyncio.CancelledError):
            self._abort(
                CANCELLED,
                original=exc,
                message="The render was cancelled",
                error_type="CancelledError",
            )
        self._abort(
            TEMPLATE_ERROR,
            original=exc,
            message=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )

    # -- frame helpers -----------------------------------------------------

    def _current_template(self) -> str | None:
        # The nearest frame with an identity is the template whose code is
        # actually executing (child override inside a base skeleton, an
        # included template or a macro's defining template).
        for frame in reversed(self._frames):
            if frame.template is not None:
                return frame.template
        return self._root_template

    @property
    def _purpose(self) -> str:
        return self._purpose_stack[-1] if self._purpose_stack else "render"

    def _make_event(
        self,
        kind: str,
        *,
        target: str | None = None,
        fragment: str | None = None,
        length: int | None = None,
        output_total: int | None = None,
        with_context: bool | None = None,
        cached: bool | None = None,
        names: tuple[str, ...] | None = None,
        binding: str | None = None,
        origin: str | None = None,
        error: dict[str, t.Any] | None = None,
        template_override: str | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            seq=self._seq,
            kind=kind,
            time=time.monotonic(),
            template=template_override or self._current_template(),
            target=target,
            depth=len(self._frames),
            is_async=self.is_async,
            purpose=self._purpose,
            fragment=fragment,
            length=length,
            output_bytes=output_total,
            with_context=with_context,
            cached=cached,
            names=names,
            binding=binding,
            origin=origin,
            error=error,
        )
        self._seq += 1
        return event

    def _dispatch(self, event: TraceEvent, *, terminal: bool) -> None:
        if self._record:
            self._events.append(event)
        if self._observer is None:
            return
        try:
            accepted = self._observer.on_event(event)
        except Exception as exc:  # noqa: BLE001 - observer supplied by user
            if terminal:
                return
            self._abort(
                REJECTED,
                original=exc,
                message=f"The trace observer raised {type(exc).__name__}: {exc}",
            )
        if not terminal and accepted is False:
            self._abort(
                REJECTED,
                message="The trace observer rejected the event",
            )

    def _emit(self, kind: str, **kwargs: t.Any) -> TraceEvent | None:
        if self._failed:
            # Late notification during exception propagation; stay quiet
            # so the original failure is not replaced.
            return None
        if self._cancel_requested or (
            self._cancel_check is not None and self._cancel_check()
        ):
            self._abort(CANCELLED, message="The render was cancelled")
        if self.max_events is not None and self._seq >= self.max_events:
            self._abort(
                EVENT_LIMIT,
                limit=self.max_events,
                message=f"Trace event limit of {self.max_events} exceeded",
            )
        event = self._make_event(kind, **kwargs)
        self._dispatch(event, terminal=False)
        return event

    def _force_emit(self, kind: str, **kwargs: t.Any) -> TraceEvent:
        event = self._make_event(kind, **kwargs)
        self._dispatch(event, terminal=True)
        return event

    def _error_state(self) -> dict[str, t.Any]:
        return {
            "seq": self._seq,
            "template": self._current_template(),
            "output_bytes": self._output_bytes,
            "output_count": self._output_count,
            "event_count": self._seq,
        }

    def _abort(
        self,
        reason: str,
        *,
        original: BaseException | None = None,
        message: str = "",
        limit: int | None = None,
        error_type: str | None = None,
    ) -> t.NoReturn:
        if self._failed and self._error is not None:
            raise self._error
        self._failed = True
        state = self._error_state()
        error_event = self._force_emit(
            RENDER_ERROR,
            target=state["template"],
            error={
                "reason": reason,
                "type": error_type or reason,
                "message": message,
            },
        )
        if reason == REJECTED:
            cls: type[RenderTraceError] = TraceRejectedError
        elif reason in (EVENT_LIMIT, OUTPUT_LIMIT):
            cls = TraceLimitError
        elif reason == TEMPLATE_ERROR:
            cls = RenderTraceError
        else:
            cls = TraceCancelledError
        error = cls(
            message,
            seq=error_event.seq,
            template=t.cast("str | None", state["template"]),
            output_bytes=t.cast(int, state["output_bytes"]),
            output_count=t.cast(int, state["output_count"]),
            event_count=t.cast(int, state["event_count"]),
            limit=limit,
            original=original,
        )
        error.reason = reason
        self._error = error
        if original is not None:
            error.__cause__ = original
        raise error

    # -- enter/exit primitives --------------------------------------------

    def _enter(
        self,
        kind: str,
        frame_kind: str,
        template: str | None = None,
        **kwargs: t.Any,
    ) -> None:
        # Stamp the event with the identity of the entered entity and
        # emit before pushing so the depth matches its exit.
        event = self._emit(kind, template_override=template, **kwargs)
        # The frame is only pushed after a successful emission; on abort
        # the render unwinds without needing the frame.
        if event is None:
            return
        self._frames.append(Frame(frame_kind, template, kwargs.get("target")))

    def _exit(self, kind: str, **kwargs: t.Any) -> None:
        # Pop first so the exit event matches the depth and identity of
        # its enter event, even when emitting failed late.
        frame = self._frames.pop() if self._frames else None
        override = kwargs.pop("template", None)
        if override is None and frame is not None:
            override = frame.template
        self._emit(kind, template_override=override, **kwargs)

    # -- compiled template hooks ------------------------------------------

    def template_enter(self, template: Template) -> None:
        identity = template_identity(template)
        self._enter(
            TEMPLATE_ENTER,
            "template",
            target=identity,
            template=identity,
        )

    def template_exit(self, template: Template) -> None:
        self._exit(TEMPLATE_EXIT, target=template_identity(template))

    def block_enter(self, template: Template, name: str) -> None:
        identity = template_identity(template)
        self._enter(BLOCK_ENTER, "block", target=name, template=identity)

    def block_exit(self, template: Template, name: str) -> None:
        self._exit(BLOCK_EXIT, target=name, template=template_identity(template))

    def extends(self, template: Template) -> None:
        self._emit(EXTENDS, target=template_identity(template))

    def include_enter(
        self, template: Template, with_context: bool, ignore_missing: bool
    ) -> None:
        identity = template_identity(template)
        self._enter(
            INCLUDE_ENTER,
            "include",
            target=identity,
            template=identity,
            with_context=with_context,
        )

    def include_exit(self, template: Template, error: bool = False) -> None:
        self._exit(
            INCLUDE_EXIT,
            target=template_identity(template),
            error={"error": True} if error else None,
        )

    def include_body_sync(
        self, template: Template, body: t.Iterable[str]
    ) -> t.Iterator[str]:
        """Iterate a cached module body for ``include ... without context``."""
        self.include_enter(template, False, False)
        try:
            yield from body
        except BaseException:
            self.include_exit(template, error=True)
            raise
        else:
            self.include_exit(template)

    async def include_body_async(
        self, template: Template, body: t.Iterable[str]
    ) -> t.AsyncIterator[str]:
        self.include_enter(template, False, False)
        try:
            # Module bodies are always materialized lists, even async, so
            # iterating synchronously is sufficient (no ``yield from`` in
            # async functions).
            for item in body:  # noqa: UP028
                yield item
        except BaseException:
            self.include_exit(template, error=True)
            raise
        else:
            self.include_exit(template)

    def extends_root_sync(
        self, template: Template, gen: t.Iterator[str]
    ) -> t.Iterator[str]:
        if getattr(template, "_trace_compiled", False):
            # The instrumented parent root emits its own template events.
            yield from gen
            return
        identity = template_identity(template)
        self._enter(TEMPLATE_ENTER, "extends", target=identity, template=identity)
        try:
            yield from gen
        except BaseException:
            self._exit(TEMPLATE_EXIT, target=identity)
            raise
        else:
            self._exit(TEMPLATE_EXIT, target=identity)

    async def extends_root_async(
        self, template: Template, gen: t.AsyncIterator[str]
    ) -> t.AsyncIterator[str]:
        if getattr(template, "_trace_compiled", False):
            async for item in gen:
                yield item
            return
        identity = template_identity(template)
        self._enter(TEMPLATE_ENTER, "extends", target=identity, template=identity)
        try:
            async for item in gen:
                yield item
        except BaseException:
            self._exit(TEMPLATE_EXIT, target=identity)
            raise
        else:
            self._exit(TEMPLATE_EXIT, target=identity)

    # -- imports -----------------------------------------------------------

    def import_module(
        self,
        template: Template,
        with_context: bool,
        context: t.Any,
        locals: dict[str, t.Any] | None,
        names: tuple[str, ...] | None = None,
        binding: str | None = None,
    ) -> t.Any:
        identity = template_identity(template)
        cached = template._trace_module_cached(with_context, context)
        self._enter(
            IMPORT_ENTER,
            "import",
            target=identity,
            template=identity,
            with_context=with_context,
            cached=cached,
            names=names,
            binding=binding,
        )
        try:
            module = template._trace_get_module(self, with_context, context, locals)
        except BaseException:
            self._exit(IMPORT_EXIT, target=identity, error={"error": True})
            raise
        self._exit(IMPORT_EXIT, target=identity)
        return module

    async def import_module_async(
        self,
        template: Template,
        with_context: bool,
        context: t.Any,
        locals: dict[str, t.Any] | None,
        names: tuple[str, ...] | None = None,
        binding: str | None = None,
    ) -> t.Any:
        identity = template_identity(template)
        cached = template._trace_module_cached(with_context, context)
        self._enter(
            IMPORT_ENTER,
            "import",
            target=identity,
            template=identity,
            with_context=with_context,
            cached=cached,
            names=names,
            binding=binding,
        )
        try:
            module = await template._trace_get_module_async(
                self, with_context, context, locals
            )
        except BaseException:
            self._exit(IMPORT_EXIT, target=identity, error={"error": True})
            raise
        self._exit(IMPORT_EXIT, target=identity)
        return module

    @contextlib.contextmanager
    def module_purpose(self, template: Template | None) -> t.Iterator[None]:
        """Mark code executed while rendering an imported template module."""
        identity = template_identity(template)
        self._purpose_stack.append("module")
        self._frames.append(Frame("module", identity, None))
        try:
            yield
        finally:
            self._frames.pop()
            self._purpose_stack.pop()

    # -- macros ------------------------------------------------------------

    def invoke_macro_sync(
        self,
        name: str | None,
        origin: str | None,
        func: t.Callable[[], str],
        autoescape: bool,
    ) -> str:
        template = origin or self._current_template()
        self._enter(MACRO_ENTER, "macro", target=name, template=template, origin=origin)
        try:
            rv = func()
        except BaseException:
            self._exit(MACRO_EXIT, target=name, template=template, origin=origin)
            raise
        self._exit(MACRO_EXIT, target=name, template=template, origin=origin)
        return Markup(rv) if autoescape else rv

    async def invoke_macro_async(
        self,
        name: str | None,
        origin: str | None,
        func: t.Callable[[], t.Awaitable[str]],
        autoescape: bool,
    ) -> str:
        template = origin or self._current_template()
        self._enter(MACRO_ENTER, "macro", target=name, template=template, origin=origin)
        try:
            rv = await func()
        except BaseException:
            self._exit(MACRO_EXIT, target=name, template=template, origin=origin)
            raise
        self._exit(MACRO_EXIT, target=name, template=template, origin=origin)
        return Markup(rv) if autoescape else rv

    # -- output ------------------------------------------------------------

    def output(self, piece: t.Any) -> t.Any:
        """Record an output fragment.

        Returns ``piece`` unchanged.  Raises a structured error when the
        output byte limit is exceeded or the session was cancelled.
        """
        text = piece if isinstance(piece, str) else str(piece)
        size = len(text.encode("utf-8"))
        if self._cancel_requested or (
            self._cancel_check is not None and self._cancel_check()
        ):
            self._abort(CANCELLED, message="The render was cancelled")
        if (
            self.max_output_bytes is not None
            and self._output_bytes + size > self.max_output_bytes
        ):
            self._abort(
                OUTPUT_LIMIT,
                limit=self.max_output_bytes,
                message=(
                    f"Trace output byte limit of {self.max_output_bytes} exceeded"
                ),
            )
        # The projected total is recorded with the event so an observer
        # sees it, but the tally only advances when the event is accepted.
        self._emit(
            OUTPUT,
            fragment=text,
            length=size,
            output_total=self._output_bytes + size,
        )
        self._output_bytes += size
        self._output_count += 1
        return piece


class _NullTrace:
    """Shared no-op session used when tracing is compiled in but inactive."""

    active = False
    events: list[TraceEvent] = []
    is_async = False
    entry = None
    failed = False
    finished = False
    max_events = None
    max_output_bytes = None

    @property
    def state(self) -> dict[str, t.Any]:
        return {
            "seq": 0,
            "template": None,
            "output_bytes": 0,
            "output_count": 0,
            "event_count": 0,
            "depth": 0,
        }

    def begin(self, *args: t.Any, **kwargs: t.Any) -> None:
        return None

    def finish(self) -> None:
        return None

    def mark_closed(self) -> None:
        return None

    def fail(self, exc: BaseException) -> t.NoReturn:
        raise exc

    def cancel(self) -> None:
        return None

    def template_enter(self, *a: t.Any, **kw: t.Any) -> None:
        return None

    def template_exit(self, *a: t.Any, **kw: t.Any) -> None:
        return None

    def block_enter(self, *a: t.Any, **kw: t.Any) -> None:
        return None

    def block_exit(self, *a: t.Any, **kw: t.Any) -> None:
        return None

    def extends(self, *a: t.Any, **kw: t.Any) -> None:
        return None

    def include_enter(self, *a: t.Any, **kw: t.Any) -> None:
        return None

    def include_exit(self, *a: t.Any, **kw: t.Any) -> None:
        return None

    def import_module(
        self,
        template: Template,
        with_context: bool = False,
        context: t.Any = None,
        locals: dict[str, t.Any] | None = None,
        names: tuple[str, ...] | None = None,
        binding: str | None = None,
    ) -> t.Any:
        return template._trace_get_module(self, with_context, context, locals)

    async def import_module_async(
        self,
        template: Template,
        with_context: bool = False,
        context: t.Any = None,
        locals: dict[str, t.Any] | None = None,
        names: tuple[str, ...] | None = None,
        binding: str | None = None,
    ) -> t.Any:
        return await template._trace_get_module_async(
            self, with_context, context, locals
        )

    @contextlib.contextmanager
    def module_purpose(self, template: t.Any) -> t.Iterator[None]:
        yield

    def invoke_macro_sync(
        self,
        name: t.Any,
        origin: t.Any,
        func: t.Callable[[], str],
        autoescape: bool,
    ) -> str:
        rv = func()
        return Markup(rv) if autoescape else rv

    async def invoke_macro_async(
        self,
        name: t.Any,
        origin: t.Any,
        func: t.Callable[[], t.Awaitable[str]],
        autoescape: bool,
    ) -> str:
        rv = await func()
        return Markup(rv) if autoescape else rv

    def output(self, piece: t.Any) -> t.Any:
        return piece

    def include_body_sync(
        self, template: t.Any, body: t.Iterable[str]
    ) -> t.Iterator[str]:
        yield from body

    async def include_body_async(
        self, template: t.Any, body: t.Iterable[str]
    ) -> t.AsyncIterator[str]:
        # Module bodies are always materialized lists, even in async mode
        # (``yield from`` is invalid inside async functions).
        for item in body:  # noqa: UP028
            yield item

    def extends_root_sync(
        self, template: t.Any, gen: t.Iterator[str]
    ) -> t.Iterator[str]:
        yield from gen

    async def extends_root_async(
        self, template: t.Any, gen: t.AsyncIterator[str]
    ) -> t.AsyncIterator[str]:
        async for item in gen:
            yield item


# A single shared instance; its methods are stateless.
null_trace = _NullTrace()
