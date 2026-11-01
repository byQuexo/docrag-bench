"""OCR with a vision LLM ("VLM"): transcribes text AND describes charts.

The second OCR engine, to compare against local RapidOCR on the same pages:

* RapidOCR reads printed characters only. A bar chart becomes loose labels
  and numbers without their meaning.
* A vision LLM can write "Figure: bar chart 'Mobile users 2014': Indonesia 64%"
  - exactly the information chart questions ask for.

The price: money per page, possible hallucinated text (and Verify would then
"verify" quotes against invented text - hence origin="vlm" on every chunk), and
pages leave the machine.

Async inside a sync interface: ingest is synchronous (and may itself run inside
the eval's event loop), while the proxy is async. So the engine owns ONE
background thread with its own event loop; ``ocr_pages`` submits a coroutine to
it and blocks until all pages are done. Pages run concurrently (bounded).
"""

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import Any

from docrag.answer.context import load_prompt
from docrag.llm.proxy import LLMProxy
from docrag.llm.types import ImagePart, Message, TextPart
from docrag.shared.concurrency import bounded_gather
from docrag.shared.schema import Origin

log = logging.getLogger(__name__)

PROMPT_VERSION = "ocr_vlm_v1"


class VlmOcrEngine:
    name = "vlm"
    origin: Origin = "vlm"
    # 1.5x is enough for a vision model and keeps image tokens (= cost) down.
    zoom = 1.5

    @staticmethod
    def version_string(route: str) -> str:
        return f"1:{route}:{PROMPT_VERSION}"

    def __init__(
        self,
        proxy: LLMProxy,
        route: str,
        *,
        concurrency: int = 8,
        temperature: float | None = None,
    ) -> None:
        self.proxy = proxy
        self.route = route
        self.concurrency = concurrency
        self.temperature = temperature
        # Route and prompt version are part of the cache key: another model or an
        # edited prompt must never reuse old transcriptions.
        self.version = self.version_string(route)
        self._prompt = load_prompt(PROMPT_VERSION)
        # LEARN: a dedicated event loop in a daemon thread. run_forever() keeps it
        # alive; run_coroutine_threadsafe() hands it work from any other thread and
        # returns a concurrent.futures.Future we can block on with .result().
        # daemon=True: the thread never keeps the process alive at exit.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def _run[T](self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def ocr_pages(self, pngs: list[bytes]) -> list[str | None]:
        return self._run(self._all(pngs))

    async def _all(self, pngs: list[bytes]) -> list[str | None]:
        gathered = await bounded_gather(
            (self._one(png) for png in pngs), limit=self.concurrency
        )
        if gathered.n_failed:
            # Failed pages come back as None: not cached, retried on the next run.
            log.warning(
                "VLM OCR: %d of %d pages failed: %s",
                gathered.n_failed,
                len(pngs),
                gathered.errors[:2],
            )
        return gathered.results

    async def _one(self, png: bytes) -> str:
        messages = [
            Message(role="system", content=self._prompt),
            Message(
                role="user",
                content=[
                    ImagePart.from_png(png),
                    TextPart(text="Transcribe this page."),
                ],
            ),
        ]
        completion = await self.proxy.chat(
            self.route, messages, temperature=self.temperature
        )
        return completion.content

    def close(self) -> None:
        self._run(self.proxy.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
