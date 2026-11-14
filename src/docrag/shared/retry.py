"""Retry with exponential backoff and full jitter, shared by the LLM and Qdrant calls.

Why jitter: when an upstream hiccups, every in-flight request fails at the same
moment. Plain exponential backoff (1 s, 2 s, 4 s) makes them all retry at the
same moment again. "Full jitter" picks a random delay in [0, 2**attempt * base],
which spreads the retries out (AWS Architecture Blog, "Exponential Backoff And
Jitter").

Why not tenacity: it is the standard library for this, and fine to use. Here
the whole policy is ~40 lines that the tests pin down exactly (injected sleep and
random), and both call sites only need "retry if transient, else raise".
"""

import asyncio
import email.utils
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime

# Never wait longer than this between two attempts, whatever Retry-After says:
# a request that has to wait a minute has already failed for the user.
MAX_DELAY_S = 20.0


def backoff_delay(
    attempt: int,
    *,
    base: float = 1.0,
    cap: float = MAX_DELAY_S,
    retry_after: float | None = None,
    rand: Callable[[], float] = random.random,
) -> float:
    """Seconds to wait before retry number ``attempt + 1`` (attempt counts from 0).

    A server-provided ``Retry-After`` wins over the computed backoff (it knows
    when its rate-limit window resets), but is capped like everything else.
    """
    if retry_after is not None:
        return min(max(retry_after, 0.0), cap)
    return min(cap, base * 2.0**attempt) * rand()


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    """``Retry-After`` as seconds: either "120" or an HTTP date. None if absent/bad."""
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max((when - datetime.now(UTC)).total_seconds(), 0.0)


def retry_sync[T](
    fn: Callable[[], T],
    *,
    retries: int,
    is_transient: Callable[[Exception], bool],
    base: float = 0.5,
    sleep: Callable[[float], None] | None = None,
) -> T:
    """Call ``fn``; on a transient error retry up to ``retries`` times."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as err:
            if attempt == retries or not is_transient(err):
                raise
        (sleep or time.sleep)(backoff_delay(attempt, base=base))
    raise AssertionError("unreachable")


async def retry_async[T](
    fn: Callable[[], Awaitable[T]],
    *,
    retries: int,
    is_transient: Callable[[Exception], bool],
    base: float = 1.0,
    retry_after: Callable[[Exception], float | None] = lambda _err: None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> T:
    """Async twin of :func:`retry_sync`, honouring a server's Retry-After."""
    for attempt in range(retries + 1):
        try:
            return await fn()
        except Exception as err:
            if attempt == retries or not is_transient(err):
                raise
            delay = backoff_delay(attempt, base=base, retry_after=retry_after(err))
        await (sleep or asyncio.sleep)(delay)
    raise AssertionError("unreachable")
