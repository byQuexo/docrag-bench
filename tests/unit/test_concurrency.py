"""bounded_gather: concurrency limit, error collection, order."""

import asyncio

from docrag.shared.concurrency import bounded_gather


async def test_limit_is_enforced() -> None:
    running = 0
    peak = 0

    async def job(i: int) -> int:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.01)  # yield so other jobs could start
        running -= 1
        return i

    result = await bounded_gather((job(i) for i in range(10)), limit=3)
    assert peak == 3
    assert result.results == list(range(10))


async def test_errors_are_collected_and_order_kept() -> None:
    async def job(i: int) -> int:
        await asyncio.sleep(0.001 * (5 - i))  # finish in reverse order
        if i == 2:
            raise ValueError("boom")
        return i * 10

    result = await bounded_gather([job(i) for i in range(5)], limit=5)
    assert result.results == [0, 10, None, 30, 40]
    assert result.n_failed == 1
    assert isinstance(result.errors[0], ValueError)


async def test_empty_input() -> None:
    result = await bounded_gather([], limit=2)
    assert result.results == []
    assert result.n_failed == 0
