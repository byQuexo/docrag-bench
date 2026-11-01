# Lesson 12: OCR for PDFs without a text layer

**Goal:** understand why a fifth of the benchmark was invisible to the system, how
we added OCR without slowing down normal PDFs, and how to compare a local OCR model
with a vision LLM fairly.

Files: `src/docrag/ingest/ocr.py`, `src/docrag/ingest/indexer.py`,
`src/docrag/ingest/chunker.py`, `src/docrag/shared/schema.py`,
`src/docrag/shared/config.py`, `src/docrag/store/qdrant.py`, `src/docrag/wiring.py`,
`src/docrag/eval/runner.py`, `tests/unit/test_ocr.py`, `scripts/make_tiny_pdf.py`,
`configs/baseline_ocr.yaml`, `configs/answer_haiku_ocr.yaml`, `pyproject.toml`.

---

## 1. The finding that started it

The first full ingest logged 28 warnings like:

```
WARNING docrag.ingest.indexer: NUS-FASS-Graduate-Guidebook-2021-small produced no chunks (scanned PDF without text layer?)
```

A quick check on one of them:

```
pages 31   text chars on page 2: 0   images on page 2: 14
```

The pages are **pictures**. Most are slide decks exported from SlideShare (file
names ending in `_95`). PyMuPDF reads the PDF's *text layer*, and these have none.

| | count |
|---|---|
| PDFs without a text layer | 28 of 134 |
| pages | 1,521 of 6,492 |
| questions affected | 255 of 1,091 (213 of 744 in the test split) |

For those questions the system could only abstain. That's a hard ceiling on every
number, and it's invisible unless you look at the logs, which is why the eval
now has a dedicated `no_text_layer` slice (section 6).

## 2. Choosing an OCR engine

| option | why (not) |
|---|---|
| **RapidOCR** (chosen first) | PaddleOCR models run on **ONNX Runtime**, the same runtime as our embedders, so it uses the GPU setup from lesson 11 and needs no PyTorch. Pip-installable, Apache-2.0. |
| Tesseract | Needs a separate system install on Windows, in Docker and in CI. Weaker on slides. |
| Docling's OCR | Pulls in PyTorch (GBs). |
| DeepSeek-OCR (open weights) | Strong, but PyTorch/vLLM, tight on 8 GB VRAM, poor Windows support. |
| Vision LLM via API (Haiku, Gemini, Mistral OCR) | Can also *describe* charts, but costs money, can hallucinate, sends pages to a third party. Built as the second engine (section 8). |

Two things checked **before** writing code, in a throwaway environment:

```powershell
uv run --isolated --no-project --with rapidocr==3.10.0 --with pymupdf --with onnxruntime python -I proto.py some.pdf
```

* `--isolated --no-project`: a temporary environment; nothing is installed into
  the project. `python -I` ignores the current directory's modules (safe when the
  inputs are downloaded files).
* Result: 70 text boxes on one slide, confidence 0.97–1.0, 1.7 s/page on CPU.
* **Supply chain:** the detection/recognition models ship *inside* the pip
  package and are hash-checked on load. So pinning `rapidocr==3.10.0` pins the
  models: no runtime downloads, same rule as the pinned Hugging Face revisions.

### A dependency surprise

After `uv add rapidocr`, both `onnxruntime` and `onnxruntime-gpu` were installed
again. `uv pip show onnxruntime` → `Required-by: pymupdf`. **PyMuPDF 2.0 depends
on the CPU onnxruntime** (for its own layout model). So the clash from lesson 11
had existed all along; GPU only worked because its files happened to be unpacked
last. The fix in `pyproject.toml`:

```toml
override-dependencies = ["onnxruntime>=1.31; extra == 'cpu'"]
```

An *override* replaces every requirement on a package, including transitive
ones. With the `extra == 'cpu'` marker, PyMuPDF's requirement only applies when
the `cpu` extra is selected. Verified with `uv export --extra cpu|gpu`: each gets
exactly one runtime. Then, because uninstalling the CPU package deleted files the
GPU package shares (same `onnxruntime/` folder), `uv sync --extra gpu
--reinstall-package onnxruntime-gpu` repaired it.

## 3. Where OCR runs: a separate pass, only where needed

OCR is **not** inside the parser. It's a second pass in `index_documents`, only
for pages the parser couldn't read:

```python
def pages_needing_ocr(pdf: bytes) -> list[int]:
    """1-based page numbers with (almost) no text layer but at least one image."""
    pages = []
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        for index, page in enumerate(doc):
            if len(page.get_text().strip()) < MIN_TEXT_CHARS and page.get_images():
                pages.append(index + 1)
    return pages
```

Why a separate pass?

* **Normal PDFs pay nothing.** The check is a cheap text-length test per page.
* **GPU in one process.** Parsing runs in 3 worker processes (CPU). Loading the
  OCR model on the GPU in each worker would load it three times and fight over
  VRAM. The OCR pass runs in the main process, model loaded once.
* **The parser stays a pure function** of the PDF bytes, so its cache is still
  valid (no parser version bump needed).

## 4. From OCR lines to Blocks

```python
for i, page in enumerate(todo):
    png = render_png(pdf, page)
    key = f"{engine.name}:{engine.version}:{OCR_ZOOM}:{bytes_hash(png)}"
    cached = cache.get_json("ocr", key) if cache is not None else None
    if cached is not None:
        text = str(cached)
    else:
        text = "\n".join(line.text for line in reading_order(engine.ocr(png)))
        if cache is not None:
            cache.set_json("ocr", key, text)
    ...
    blocks.append(
        Block(
            ...,
            text=text,
            page=page,
            section_path=[],
            order=first_order + len(blocks),
            origin="ocr",
        )
    )
```

Step by step:

1. **Render** the page at 2x zoom (144 dpi): small slide text becomes readable.
2. **Cache key** = engine name + engine version + zoom + hash of the *image*. Same
   rule as everywhere: everything that changes the output is in the key. Zoom is
   in it because a different resolution gives different OCR text.
3. **Reading order:** OCR returns boxes in detection order. `reading_order()`
   buckets the y coordinate (boxes on one visual line differ by a few pixels) and
   sorts by (row, x). Known limitation: two-column slides are read row by row.
4. **One Block per page**, `origin="ocr"`, no heading chain (OCR doesn't know
   which line is a heading).

### `origin` travels through the whole pipeline

```python
# shared/schema.py
Origin = Literal["text", "ocr", "vlm"]
...
origin: Origin = "text"
```

Block → Chunk → Qdrant payload → Hit → (later) citations. The default `"text"`
means chunks indexed before OCR existed still load. That's **backward-compatible
schema evolution**: add fields with defaults, never repurpose old ones.

### The chunker needed one change

OCR blocks all have `section_path == []`. The chunker merges consecutive text of
the *same section* up to 200 words, so it would have glued slide 1, 2 and 3
into one chunk spanning three pages, and page-level citations would point to the
wrong slide. Now the merge key includes origin and, for OCR text, the page:

```python
page_key = block.page if block.origin != "text" else None
here = (block.doc_id, block.section_path, block.origin, page_key)
```

One slide = one chunk. Test: `test_ocr_text_is_chunked_per_page_never_glued_across_slides`.

## 5. OCR is an experiment axis, with its own collection

```yaml
# configs/baseline_ocr.yaml
ocr: rapidocr            # OCR pages without a text layer (local, GPU)
```

```python
def collection_name(embedder, dim, prefix_headings, ocr="none"):
    name = f"chunks__{safe}__{dim}__{'pfx' if prefix_headings else 'raw'}"
    return name if ocr == "none" else f"{name}__ocr-{ocr}"
```

OCR changes *which chunks exist*, so it must not write into the non-OCR
collection. Otherwise "baseline" would silently become "baseline + OCR" and every
old result would be incomparable. `ocr="none"` keeps the old name, so existing
collections stay valid.

The rerank threshold is **recalibrated** for the OCR variant on dev, since the OCR
text changes the score distribution.

## 6. Measuring it: the `no_text_layer` slice

`eval/runner.py` tags each question whose document has no text layer (the cached
parse returned no blocks) with `no_text_layer`, so every metric is reported for that
slice too. While adding it, a bug surfaced:

```python
# before: substring match - "text" also matched "no_text_layer"!
df.filter(pl.col("slices").str.contains(name))
# after: exact tag match
df.filter(pl.col("slices").str.split(",").list.contains(name))
```

## 7. Results

Test split, 735 questions, same pipeline (hybrid + rerank, gte-small), only `ocr`
differs. RapidOCR on the GPU processed ~1,100 test pages in about 3 minutes.

| | recall@10 all | MRR@10 all | recall@10 `no_text_layer` (n=213) | correct all | correct `no_text_layer` |
|---|---|---|---|---|---|
| no OCR | 0.550 | 0.412 | 0.000 | 0.392 | 0.282 |
| RapidOCR | **0.760** | **0.607** | **0.776** | **0.464** | **0.526** |

How to read it:

* **+21 points recall@10 overall**: the biggest single improvement in the project.
  Bigger than hybrid search, reranking or a 2x larger embedding model combined.
* `no_text_layer` correct without OCR is 0.282, not 0: those are the unanswerable
  questions on image-only documents, where abstaining is correct.
* **Faithfulness does NOT drop** (0.458 → 0.461 over judged answers). The first
  version of this lesson said "0.842 → 0.782, because OCR text is noisier". That
  was a metric bug: every abstention counted as faithful, and OCR simply produced
  fewer abstentions (0.693 → 0.574), so the average fell. A plausible story
  explained an artifact. Fixed on 2026-11-01: `faithful` is now averaged only over
  answers the judge model read (`judge.judged_by_llm`).
* Citation precision rises (0.67 → 0.73): OCR'd slides are short, one per chunk, so
  a citation points at exactly one page.

The lesson for interviews: **look at the data before tuning models.** Swapping the
embedding model (384 → 768 dims) changed recall@10 by 0.002; reading the ingest
warnings and fixing the input changed it by 0.21.

## 8. The vision-LLM engine

RapidOCR reads characters. On a chart slide that gives a flat list:

```
UCWeb | Telecom Operators - late 2014 (4) | Subscribers (in millions) | 139.3 | ... | 63.5 | 58.3 | 54.2 | 35.4 | 32 | 29 | ... | Telkomsel | XL | Indosat
```

Which number belongs to which operator? The layout is lost. On another slide it
read "30%" as "3096". The vision LLM (Claude Haiku 5.5, same slide):

```
Figure: grouped bar chart 'Telecom Operators – late 2014 (4)' ...
Telkomsel: Subscribers 139.3, Smartphone 35.4, BlackBerry Users 10.4, Android 17.3, Data Users 63.5
XL: Subscribers 58.3, Smartphone 15, BlackBerry Users 3, Android 8, Data Users 32
```

Numbers bound to their labels: exactly what "How many data users did XL have?"
needs. Cost measured on real slides: **~0.00017 EUR per text slide, ~0.0004 EUR per
chart slide** (OpenRouter-reported).

### 8.1 Images in the message format (`llm/types.py`)

Until now `Message.content` was a `str`. A vision request needs text *and* images:

```python
class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImagePart(BaseModel):
    type: Literal["image"] = "image"
    media_type: Literal["image/png", "image/jpeg"] = "image/png"
    data: str  # base64, no "data:" prefix


ContentPart = Annotated[TextPart | ImagePart, Field(discriminator="type")]


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]

    @property
    def text(self) -> str:
        """The text of the message, ignoring images."""
```

* A **discriminated union**: Pydantic reads `type` to decide which class a dict
  becomes. TS: a union narrowed on `kind`. Rust: an enum with two variants.
* `content` stays `str | list[...]`, so every existing caller (answers, judge,
  threads) keeps sending plain strings. **Widen a type, don't break it.**
* `.text` gives code that only needs text (Anthropic's `system` field, the fake
  provider) one place to read it.

### 8.2 Each provider converts images its own way

The whole point of the provider adapters (lesson 05): the same `ImagePart`
becomes two different wire formats.

```python
# llm/openai_compat.py: OpenAI format, also used by OpenRouter
{
    "type": "image_url",
    "image_url": {"url": f"data:{part.media_type};base64,{part.data}"},
}

# llm/anthropic_provider.py: Anthropic Messages API
{
    "type": "image",
    "source": {"type": "base64", "media_type": part.media_type, "data": part.data},
}
```

Both are pinned by tests in `tests/unit/test_vlm_ocr.py`.

### 8.3 Async API, sync ingest: a loop in a thread (`ingest/vlm_ocr.py`)

Ingest is synchronous code, and the eval calls it from *inside* an `async`
function. The proxy is async. `asyncio.run()` would fail there ("cannot be called
from a running event loop"). The engine owns its own loop in a background thread:

```python
self._loop = asyncio.new_event_loop()
self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
self._thread.start()


def _run[T](self, coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run_coroutine_threadsafe(coro, self._loop).result()
```

* `run_forever()` keeps the loop alive in the thread.
* `run_coroutine_threadsafe()` hands it a coroutine from any other thread and
  returns a `concurrent.futures.Future`; `.result()` blocks until it's done.
* `daemon=True`: the thread never keeps the process alive at exit.
* The httpx client inside the proxy is only ever used from this one loop, which
  matters: async HTTP clients are bound to the loop they first ran on.

Java analogy: an `ExecutorService` you submit `Callable`s to and `get()` the
`Future`. Rust: `tokio::runtime::Runtime::block_on` from sync code.

### 8.4 Batch pages, run them concurrently

The engine interface changed from `ocr(png)` to `ocr_pages(pngs)`:

```python
def ocr_pages(self, pngs: list[bytes]) -> list[str | None]:
    """Text per page image, same order. None = failed (not cached, retried)."""
```

* RapidOCR just loops (local GPU, ~0.2 s/page).
* The VLM engine sends up to 8 pages at once with `bounded_gather` (lesson 01).
  1,500 pages × ~4 s sequentially would be 1.5 hours; 8 at a time is ~12 minutes.
* `None` for a failed page: `ocr_blocks` caches only successes, so a rate-limit
  hiccup is retried on the next run instead of being remembered as an empty
  page. Test: `test_failed_pages_are_not_cached_and_retried_next_time`.

### 8.5 What goes into the cache key and the collection name

```python
self.version = f"1:{route}:{PROMPT_VERSION}"  # engine version
key = f"{engine.name}:{engine.version}:{engine.zoom}:{bytes_hash(png)}"
```

Another model, another prompt or another zoom: a new key. And the Qdrant collection
name includes the model, `...__ocr-vlm-openrouter-anthropic-claude-haiku-5.5`, so
two vision models can be compared side by side.

### 8.6 The trust problem

Verify checks that a quote appears in the chunk. For VLM chunks, the chunk text is
itself model output. If the model wrote "Android: 2013 60%" but the chart says
55%, an answer quoting "60%" passes Verify. Mitigations in this project:

* `origin="vlm"` on every block and chunk, so the eval can slice by it;
* the prompt forbids inventing text and asks for `[unreadable]` instead;
* the comparison with RapidOCR on the same pages (section 9).

What a production system would add: show the page image next to every VLM
citation (`/pages/{doc}/{page}.png` already exists), and spot-check a sample of
transcriptions against the images.

## 9. RapidOCR vs vision LLM

Same pipeline, same test questions, only the OCR engine differs. Each variant has
its own collection and its own threshold calibrated on dev (0.577 vs 0.632).

| | recall@10 all | recall@10 `no_text_layer` | correct all | correct chart/figure | correct `no_text_layer` | faithful | cost |
|---|---|---|---|---|---|---|---|
| no OCR | 0.550 | 0.000 | 0.392 | 0.102 | 0.282 | 0.458 | 0 |
| RapidOCR (local GPU) | 0.760 | 0.776 | 0.464 | 0.213 | 0.526 | 0.461 | 0 EUR, ~3 min |
| Vision LLM (Haiku 5.5) | **0.776** | **0.835** | **0.501** | **0.306** | **0.653** | 0.495 | 0.42 EUR for 1,390 pages, ~12 min |

How to read it:

* **Most of the gain comes from OCR at all** (+21 pts recall@10). The vision LLM
  adds another +1.6 overall and +6 on image-only documents.
* **The answer quality gap is bigger than the retrieval gap**: on charts, correct
  goes 0.213 → 0.306 (+44% relative). Retrieval finds the right slide with both
  engines; only the VLM text says which number belongs to which bar.
* **Faithfulness doesn't drop** with the VLM (0.461 vs 0.495, judged answers only). There is no
  sign of widespread hallucination at this level. But the judge only checks the
  answer against the *chunk text*, not against the image, so a hallucinated
  transcription would be invisible to it. The honest statement: "no measured
  harm, not proven absent".
* **Cost-benefit:** ~0.0003 EUR per page. For a one-off corpus that's nothing; for
  millions of pages it's real money and latency, where RapidOCR + VLM only for
  chart pages would be the next experiment.
* **A gap this run revealed:** the OCR proxy's spend isn't in `run.json` (it has
  its own `LLMProxy`). The 0.42 EUR was reconstructed from the cache, where every
  completion stores its billed `cost_usd`. Fixing that is exercise 5.

---

## Python concepts in this lesson

* **Structural typing for engines.** `OcrEngine` is a `Protocol`; the test's
  `FakeOcrEngine` satisfies it without inheriting anything (TS interface style).
* **Lazy imports for optional heavy dependencies.** `import rapidocr` lives
  inside `RapidOcrEngine.__init__`, so importing `docrag.ingest.ocr` doesn't load
  OpenCV. (Java: loading a class only when first used; Rust: an optional feature.)
* **`typing.Any` as a deliberate escape hatch.** RapidOCR returns a union of four
  result types that mypy can't narrow; `result: Any = self._engine(png)` documents
  "we know which one this is" for exactly one value.
* **`importlib.metadata.version("rapidocr")`** reads the installed package
  version, the reliable way when a package has no `__version__`.
* **`pytest.importorskip("rapidocr")`** skips a test when an optional dependency
  is missing instead of failing it.
* **`filter(None, [...])`** drops empty strings before `",".join(...)`.

## Exercises

1. **Zoom vs accuracy.** Change `OCR_ZOOM` to 1.0, re-run
   `uv run pytest tests/unit/test_ocr.py -k rapidocr`. *Done when* you can say
   whether the test still passes and why zoom is in the cache key.
2. **Find a partly scanned PDF.** Write a 5-line script that prints, for every PDF
   in `data/mmlongbench/documents`, how many pages need OCR and how many pages it
   has. *Done when* you've found a PDF that has a text layer on some pages but not
   others (the OCR pass covers those too).
3. **Reading order.** Add a test for a two-column slide where `reading_order`
   gives the "wrong" order. *Done when* the test documents the limitation (mark it
   `xfail` with a reason).
4. **Origin in the API.** Add `origin` to `SourceOut` in `api/app.py` so a UI
   could show "read by OCR" next to a citation. *Done when* `test_api.py` passes.
5. **Report the OCR cost.** `VlmOcrEngine` has its own `LLMProxy`, so its spend is
   missing from `run.json`. Make `index_documents` return the engine (or its
   `proxy.stats`) and write `ocr_cost_eur` into `run.json`. *Done when* a
   `baseline_vlm` re-run (all cached, 0 EUR) shows `ocr_cost_eur: 0.0` and a fresh
   page shows a non-zero cost.
6. **Hallucination spot-check.** Pick 10 chart slides, compare the VLM's
   "Figure:" numbers with the page image (`/pages/{doc}/{page}.png` or
   `render_png`). *Done when* you can state an error rate, which is an
   interview-ready number.

## Interview questions

**Q: Why did a fifth of your benchmark score zero, and how did you notice?**
28 of 134 PDFs were image-only slide decks; the text parser produced no chunks,
logged as warnings during ingest. Counting the affected questions (255 of 1,091)
turned a log line into a number, and the eval now has a `no_text_layer` slice so
it can't hide in an average again.

**Q: Why RapidOCR?**
It runs PaddleOCR models on ONNX Runtime, which the project already used for
embeddings and reranking, so it reuses the GPU setup and adds no PyTorch. It's a
pip package (no system install in CI/Docker), and its models ship inside the
pinned package, so no runtime downloads.

**Q: Why is OCR a separate pass instead of part of the parser?**
Only pages without a text layer need it, so normal PDFs pay nothing. The GPU
model loads once in the main process instead of once per parser worker. And the
parser stays a pure function of the PDF, so its cache remains valid.

**Q: What can go wrong with OCR text in a RAG system with verified citations?**
Verify checks that a quote appears in the chunk text, but for OCR the chunk text is
itself a model's output. If OCR misreads "45.5" as "46.5", a quote of "46.5" is
"verified". That's why every chunk carries `origin` and the eval reports OCR
documents separately.

**Q: Why does the OCR variant get its own Qdrant collection?**
It changes which chunks exist. Writing into the baseline collection would turn
the baseline into "baseline + OCR" without anyone noticing, and every earlier
result would become incomparable. A separate collection makes OCR an explicit
experiment axis.

**Q: RapidOCR or a vision LLM, which would you ship?**
Both, routed. RapidOCR is free, local and deterministic and gets most of the gain
(recall@10 0.55 → 0.76). The vision LLM adds the chart understanding (correct on
chart questions 0.21 → 0.31) for ~0.0003 EUR/page. In production I'd run RapidOCR
everywhere and send only pages with detected charts to the VLM, show the page
image next to VLM-derived citations, and spot-check transcriptions.

**Q: How do you know the vision LLM didn't hallucinate?**
I don't fully. Faithfulness stayed at the RapidOCR level, but the judge compares
against chunk text, not the image. Every chunk carries `origin="vlm"`, the prompt
forbids invention, and the next step is a sampled manual check of chart numbers
against page images to get a real error rate.

**Q: What else did adding OCR uncover?**
PyMuPDF 2.0 depends on the CPU onnxruntime, so the GPU/CPU runtime clash had been
there all along. Fixed with a uv override scoped to the `cpu` extra. Also a
substring-matching bug in the slice filter ("text" matched "no_text_layer").
