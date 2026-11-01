"""A @timed decorator: latency p50/p95 for the results table + an OTel span.

The plan said "Timing decorator before OpenTelemetry". That paid off: when
tracing arrived (M6), only this file changed - every timed function was already
marked, so every one of them became a span without touching a call site.

Two outputs, on purpose:
* the in-memory registry below -> run.json latency percentiles (always on, works
  without any collector, is what the benchmark tables report);
* an OpenTelemetry span -> per-request breakdown in Phoenix (only inside a trace,
  see shared/tracing.py).
"""

import functools
import inspect
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar

import numpy as np

from docrag.shared import tracing

P = ParamSpec("P")
R = TypeVar("R")

# name -> list of durations in seconds. Module-level = one per process.
_DURATIONS: defaultdict[str, list[float]] = defaultdict(list)


# The flag of the innermost running @timed call (None outside of one).
# LEARN: a ContextVar, not a plain global: with asyncio, many timed calls run
# interleaved on one thread, and each must only ever see its own flag.
_CACHE_HIT: ContextVar[list[bool] | None] = ContextVar("docrag_cache_hit", default=None)


def record(name: str, seconds: float) -> None:
    _DURATIONS[name].append(seconds)


def cache_hit() -> None:
    """Mark the current @timed call as answered from the cache.

    Its duration is then recorded as ``<name>_cached`` instead of ``<name>``.
    Without this, a re-run where 99% of LLM calls are cache hits reported
    "llm p50 = 3 ms": the percentile of cache lookups, not of the LLM.
    """
    flag = _CACHE_HIT.get()
    if flag is not None:
        flag[0] = True
    tracing.annotate({"docrag.cache_hit": True})


@contextmanager
def _span(name: str, kind: str | None, root: bool) -> Iterator[None]:
    """A span named ``name`` - but only inside a trace, unless ``root``.

    "Only inside a trace" is what keeps ingest out of Phoenix: parse/embed/upsert
    during indexing run without an active trace, so they cost one contextvar
    lookup and record nothing. The same embed() inside a query becomes a child
    span of that query.
    """
    if not (root or tracing.in_trace()):
        yield
        return
    attrs = {tracing.SPAN_KIND: kind} if kind else None
    with tracing.tracer().start_as_current_span(name, attributes=attrs):
        yield


def timed(
    name: str, *, kind: str | None = None, root: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that records how long each call takes under ``name``.

    Also opens an OpenTelemetry span of that name when called inside a trace.
    ``kind``: OpenInference span kind (tracing.LLM, tracing.RETRIEVER, ...).
    ``root=True``: may START a trace (used by the answer entry point), instead of
    only joining one. Works for normal functions and ``async def`` functions.
    """

    # LEARN: A decorator with an argument is a function returning a decorator:
    #   @timed("embed")  ==  func = timed("embed")(func)
    # ParamSpec `P` keeps the wrapped function's exact signature for mypy, so
    # wrapper(*args, **kwargs) is still type-checked at call sites.
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Any:
                with _span(name, kind, root):
                    flag = [False]
                    token = _CACHE_HIT.set(flag)
                    start = time.perf_counter()
                    try:
                        return await func(*args, **kwargs)
                    finally:
                        _CACHE_HIT.reset(token)
                        record(_timer_name(name, flag), time.perf_counter() - start)

            return async_wrapper  # type: ignore[return-value]

        # LEARN: functools.wraps copies __name__, __doc__ etc. from func to wrapper,
        # otherwise every decorated function would show up as "wrapper" in traces.
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # perf_counter: monotonic, high resolution. Never use time.time() to
            # measure durations (it jumps when the system clock is adjusted).
            with _span(name, kind, root):
                # A fresh flag per call; reset() restores the caller's flag, so a
                # cache hit in a nested timed call never marks its parent.
                flag = [False]
                token = _CACHE_HIT.set(flag)
                start = time.perf_counter()
                try:
                    return func(*args, **kwargs)
                finally:
                    _CACHE_HIT.reset(token)
                    record(_timer_name(name, flag), time.perf_counter() - start)

        return wrapper

    return decorator


def _timer_name(name: str, flag: list[bool]) -> str:
    return f"{name}_cached" if flag[0] else name


def percentiles(name: str) -> dict[str, float]:
    """p50/p95 in milliseconds for one timer name (empty dict if never called)."""
    values = _DURATIONS.get(name)
    if not values:
        return {}
    arr = np.asarray(values) * 1000.0
    return {
        "n": float(len(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def snapshot() -> dict[str, dict[str, float]]:
    return {name: percentiles(name) for name in sorted(_DURATIONS)}


def reset() -> None:
    _DURATIONS.clear()
