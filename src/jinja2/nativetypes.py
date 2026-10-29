import typing as t
from ast import literal_eval
from ast import parse
from itertools import chain
from itertools import islice
from types import GeneratorType

from . import nodes
from .compiler import CodeGenerator
from .compiler import Frame
from .compiler import has_safe_repr
from .environment import Environment
from .environment import Template


def native_concat(values: t.Iterable[t.Any]) -> t.Any | None:
    """Return a native Python type from the list of compiled nodes. If
    the result is a single node, its value is returned. Otherwise, the
    nodes are concatenated as strings. If the result can be parsed with
    :func:`ast.literal_eval`, the parsed value is returned. Otherwise,
    the string is returned.

    :param values: Iterable of outputs to concatenate.
    """
    head = list(islice(values, 2))

    if not head:
        return None

    if len(head) == 1:
        raw = head[0]
        if not isinstance(raw, str):
            return raw
    else:
        if isinstance(values, GeneratorType):
            values = chain(head, values)
        raw = "".join([str(v) for v in values])

    try:
        return literal_eval(
            # In Python 3.10+ ast.literal_eval removes leading spaces/tabs
            # from the given string. For backwards compatibility we need to
            # parse the string ourselves without removing leading spaces/tabs.
            parse(raw, mode="eval")
        )
    except (ValueError, SyntaxError, MemoryError):
        return raw


class NativeCodeGenerator(CodeGenerator):
    """A code generator which renders Python types by not adding
    ``str()`` around output nodes.
    """

    @staticmethod
    def _default_finalize(value: t.Any) -> t.Any:
        return value

    def _output_const_repr(self, group: t.Iterable[t.Any]) -> str:
        return repr("".join([str(v) for v in group]))

    def _output_child_to_const(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> t.Any:
        const = node.as_const(frame.eval_ctx)

        if not has_safe_repr(const):
            raise nodes.Impossible()

        if isinstance(node, nodes.TemplateData):
            return const

        return finalize.const(const)  # type: ignore

    def _output_child_pre(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> None:
        if finalize.src is not None:
            self.write(finalize.src)

    def _output_child_post(
        self, node: nodes.Expr, frame: Frame, finalize: CodeGenerator._FinalizeInfo
    ) -> None:
        if finalize.src is not None:
            self.write(")")


class NativeEnvironment(Environment):
    """An environment that renders templates to native Python types."""

    code_generator_class = NativeCodeGenerator
    concat = staticmethod(native_concat)  # type: ignore


class NativeTemplate(Template):
    environment_class = NativeEnvironment

    def render(self, *args: t.Any, trace: t.Any = None, **kwargs: t.Any) -> t.Any:
        """Render the template to produce a native Python type. If the
        result is a single node, its value is returned. Otherwise, the
        nodes are concatenated as strings. If the result can be parsed
        with :func:`ast.literal_eval`, the parsed value is returned.
        Otherwise, the string is returned.
        """
        session = self._prepare_trace(trace)
        ctx = self.new_context(dict(*args, **kwargs), trace=session)

        if session is None:
            with self._trace_isolation(None):
                try:
                    return self.environment_class.concat(  # type: ignore
                        self.root_render_func(ctx)
                    )
                except Exception:
                    return self.environment.handle_exception()

        session.begin("render", self, is_async=False)
        gen = t.cast("t.Generator[t.Any, None, None]", self.root_render_func(ctx))

        def traced() -> t.Iterator[t.Any]:
            with session.activate():
                try:
                    for piece in gen:
                        session.output(piece)
                        yield piece
                except GeneratorExit:
                    session.mark_closed()
                    raise

        try:
            return self.environment_class.concat(traced())  # type: ignore
        except BaseException as exc:
            self._safe_close_sync(gen)
            raise session.fail(exc) from exc
        finally:
            session.finish()

    async def render_async(
        self, *args: t.Any, trace: t.Any = None, **kwargs: t.Any
    ) -> t.Any:
        if not self.environment.is_async:
            raise RuntimeError(
                "The environment was not created with async mode enabled."
            )

        session = self._prepare_trace(trace)
        ctx = self.new_context(dict(*args, **kwargs), trace=session)

        if session is None:
            with self._trace_isolation(None):
                try:
                    return self.environment_class.concat(  # type: ignore
                        [n async for n in self.root_render_func(ctx)]  # type: ignore
                    )
                except Exception:
                    return self.environment.handle_exception()

        import asyncio

        session.begin("render", self, is_async=True)
        agen: t.AsyncGenerator[t.Any, None] = self.root_render_func(ctx)  # type: ignore
        pieces: list[t.Any] = []
        try:
            with session.activate():
                async for piece in agen:
                    session.output(piece)
                    pieces.append(piece)
            return self.environment_class.concat(pieces)  # type: ignore
        except (Exception, asyncio.CancelledError) as exc:
            await self._safe_aclose(agen)
            raise session.fail(exc) from exc
        finally:
            session.finish()


NativeEnvironment.template_class = NativeTemplate
