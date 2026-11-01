"""bounded_gather: run many coroutines, but at most N at the same time.

Used for LLM calls during eval (answers + judge). Without a bound, 1000 questions
would fire 1000 HTTP requests at once and hit rate limits immediately.
"""

import asyncio
from collections.abc import Awaitable, Iterable
from dataclasses import dataclass, field


@dataclass
class GatherResult[T]:
    # LEARN: `class GatherResult[T]` is Python 3.12 generic syntax, the same idea as
    # `class GatherResult<T>` in Java/TS. Older code writes `T = TypeVar("T")` plus
    # `class X(Generic[T])`; you'll read both styles in the wild.
    # @dataclass auto-generates __init__, __repr__, __eq__ from the fields, like a
    # Java record / Kotlin data class. Use dataclasses for plain internal data and
    # Pydantic models where you need validation or JSON (API, LLM output, config).
    results: list[T | None]  # same order as the input; None where it failed
    errors: list[BaseException] = field(default_factory=list)

    @property
    def n_failed(self) -> int:
        return len(self.errors)


async def bounded_gather[T](
    coros: Iterable[Awaitable[T]], limit: int = 4
) -> GatherResult[T]:
    """Await all ``coros`` with at most ``limit`` running concurrently.

    Errors don't cancel the other tasks (``return_exceptions=True``); they are
    collected and counted so the eval report can say "3 of 500 calls failed"
    instead of crashing after an hour.
    """
    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Awaitable[T]) -> T:
        # LEARN: `async with semaphore:` acquires one of `limit` permits and releases
        # it on exit, even on exceptions (like try/finally or Rust's Drop guard).
        async with semaphore:
            return await coro

    # LEARN: asyncio.gather is roughly Promise.allSettled when
    # return_exceptions=True: exceptions come back as values instead of being
    # raised. Order of results == order of inputs, regardless of finish order.
    raw = await asyncio.gather(*(run(c) for c in coros), return_exceptions=True)

    out = GatherResult[T](results=[])
    for item in raw:
        if isinstance(item, BaseException):
            out.errors.append(item)
            out.results.append(None)
        else:
            out.results.append(item)
    return out
