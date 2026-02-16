"""Decorators for automatic span creation around eval functions."""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Callable, TypeVar

from tracing.otel_setup import get_tracer

F = TypeVar("F", bound=Callable[..., Any])


def trace_eval(func: F) -> F:
    """Decorator that wraps *func* in an OTel span.

    Works transparently for both sync and async callables.  If OTel is not
    installed (``get_tracer()`` returns ``None``) the original function is
    executed without any tracing overhead.

    Span attributes set:
    * ``eval.function``    — qualified name of the wrapped function
    * ``eval.duration_ms`` — wall-clock duration in milliseconds
    * ``eval.status``      — ``"ok"`` or ``"error"``
    """

    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            if tracer is None:
                return await func(*args, **kwargs)

            with tracer.start_as_current_span(func.__qualname__) as span:
                span.set_attribute("eval.function", func.__qualname__)
                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("eval.status", "ok")
                    return result
                except Exception as exc:
                    span.set_attribute("eval.status", "error")
                    span.record_exception(exc)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    span.set_attribute("eval.duration_ms", duration_ms)

        return async_wrapper  # type: ignore[return-value]

    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            tracer = get_tracer()
            if tracer is None:
                return func(*args, **kwargs)

            with tracer.start_as_current_span(func.__qualname__) as span:
                span.set_attribute("eval.function", func.__qualname__)
                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("eval.status", "ok")
                    return result
                except Exception as exc:
                    span.set_attribute("eval.status", "error")
                    span.record_exception(exc)
                    raise
                finally:
                    duration_ms = (time.perf_counter() - start) * 1000
                    span.set_attribute("eval.duration_ms", duration_ms)

        return sync_wrapper  # type: ignore[return-value]
