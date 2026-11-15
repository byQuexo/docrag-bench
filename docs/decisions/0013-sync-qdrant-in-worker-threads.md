# 0013. Sync Qdrant client in worker threads, async only for LLM calls

- **Status:** Accepted
- **Date:** 2026-11-01 (`asyncio.to_thread` for retrieval, in `e1ca9c0`); kept when adding retries and probes, 2026-11-14 (`b5d68d8`, `6fe9283`)

## Context

FastAPI runs `async def` handlers on one event loop. Retrieval is one
function: embed (ONNX, CPU) → search (Qdrant, I/O) → rerank (ONNX, CPU). A
300 ms CPU call inside the handler would freeze every other request for 300 ms.

## Decision

`AnswerService.answer` runs the whole retrieval in a worker thread with
`asyncio.to_thread`; the Qdrant search rides along with the sync
`QdrantClient`. The LLM calls, which wait seconds, are native async (httpx,
`AsyncAnthropic`). `/readyz` runs its Qdrant ping the same way, under
`asyncio.wait_for`.

## Alternatives considered

- **`AsyncQdrantClient`:** the CPU parts must leave the event loop anyway, so
  the function would be split in three (thread, await, thread) for no
  measurable gain.
- **A process pool for the models:** more isolation, but the models would load
  once per process; ONNX Runtime and numpy already release the GIL while computing.

## Consequences

- **+** One simple retrieval function, used unchanged by the API, CLI and eval.
- **+** Other requests keep being served while one is embedding or reranking.
- **−** Concurrency is bounded by the default thread pool, not by the event loop.
- **−** A hanging sync call can only be abandoned, not cancelled: the request
  deadline ([0014](0014-retries-and-deadlines.md)) answers 504, but the thread
  runs until the Qdrant client's own timeout.

## Evidence

- `src/docrag/answer/service.py`, `src/docrag/store/qdrant.py`, `src/docrag/api/app.py` (`/readyz`)
- [Lesson 07, section 6](../lessons/07_api_and_threads.md), [lesson 14, section 4](../lessons/14_operability.md)
