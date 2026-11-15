"""Retry policy: jittered backoff, Retry-After, transient vs permanent errors."""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from docrag.shared.retry import (
    MAX_DELAY_S,
    backoff_delay,
    parse_retry_after,
    retry_async,
    retry_sync,
)
from docrag.store.qdrant import is_transient


@pytest.mark.parametrize("attempt", [0, 1, 2, 3])
def test_full_jitter_stays_within_the_exponential_bound(attempt: int) -> None:
    assert backoff_delay(attempt, base=1.0, rand=lambda: 0.0) == 0.0
    assert backoff_delay(attempt, base=1.0, rand=lambda: 1.0) == 2**attempt


def test_backoff_is_capped() -> None:
    assert backoff_delay(30, rand=lambda: 1.0) == MAX_DELAY_S


def test_retry_after_wins_but_is_capped() -> None:
    assert backoff_delay(0, retry_after=3.0) == 3.0
    assert backoff_delay(0, retry_after=3600.0) == MAX_DELAY_S


def test_parse_retry_after() -> None:
    assert parse_retry_after({"retry-after": "7"}) == 7.0
    assert parse_retry_after({}) is None
    assert parse_retry_after({"retry-after": "soon"}) is None
    later = format_datetime(datetime.now(UTC) + timedelta(seconds=30), usegmt=True)
    assert 25 < (parse_retry_after({"retry-after": later}) or 0) <= 30


def test_retry_sync_retries_transient_errors_then_succeeds() -> None:
    calls: list[int] = []
    sleeps: list[float] = []

    def flaky() -> str:
        calls.append(1)
        if len(calls) < 3:
            raise TimeoutError
        return "ok"

    out = retry_sync(
        flaky,
        retries=2,
        is_transient=lambda e: isinstance(e, TimeoutError),
        sleep=sleeps.append,
    )
    assert out == "ok"
    assert len(calls) == 3 and len(sleeps) == 2


def test_retry_sync_gives_up_after_the_last_retry() -> None:
    calls: list[int] = []

    def down() -> None:
        calls.append(1)
        raise TimeoutError

    with pytest.raises(TimeoutError):
        retry_sync(down, retries=2, is_transient=lambda e: True, sleep=lambda s: None)
    assert len(calls) == 3


def test_permanent_errors_are_not_retried() -> None:
    calls: list[int] = []

    def bad() -> None:
        calls.append(1)
        raise ValueError("wrong vector size")

    with pytest.raises(ValueError, match="wrong vector size"):
        retry_sync(bad, retries=5, is_transient=lambda e: False, sleep=lambda s: None)
    assert len(calls) == 1


async def test_retry_async_passes_retry_after_to_sleep() -> None:
    sleeps: list[float] = []

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)

    attempts = iter([RuntimeError("429"), None])

    async def call() -> str:
        err = next(attempts)
        if err is not None:
            raise err
        return "ok"

    out = await retry_async(
        call,
        retries=1,
        is_transient=lambda e: True,
        retry_after=lambda e: 4.0,
        sleep=fake_sleep,
    )
    assert out == "ok" and sleeps == [4.0]


def _unexpected(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(status, "x", b"", httpx.Headers())


@pytest.mark.parametrize(
    ("err", "transient"),
    [
        (ResponseHandlingException(ConnectionError()), True),
        (_unexpected(503), True),
        (_unexpected(429), True),
        (_unexpected(404), False),
        (_unexpected(400), False),
        (ValueError(), False),
    ],
)
def test_qdrant_transient_classification(err: Exception, transient: bool) -> None:
    assert is_transient(err) is transient
