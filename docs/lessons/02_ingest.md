# Lesson 02: Ingest — PDF → Blocks → Chunks (`src/docrag/ingest/`)

**Goal:** understand how a real PDF becomes searchable, citable chunks that know
their page and heading chain, and how the work is cached and parallelized.
Milestone M1: "tiny.pdf yields chunks; a table stays one chunk; every chunk has a page".

## What happens step by step

```
PDF bytes ──parser.py──▶ list[Block] ──chunker.py──▶ list[Chunk] ──indexer.py──▶ Qdrant
             (cached by content hash)                 (stable IDs)    (parallel parse, embed, upsert)
```

### 1. The parser contract (`parser.py`)

```python
class Parser(Protocol):
    name: str
    version: str  # goes into the cache key

    def parse(self, pdf: bytes, doc_id: str) -> list[Block]: ...
```

Plus a tiny registry (`register("pymupdf", PyMuPdfParser)` / `get_parser("pymupdf")`).
v1 has exactly one implementation. The rule from the plan: a second parser only
when a benchmark justifies it.

### 2. `PyMuPdfParser.parse()`: how a page becomes Blocks

1. `pymupdf.open(stream=pdf, filetype="pdf")` inside a `with` block → the document
   is always closed.
2. `page.get_text("dict")` for every page: blocks → lines → spans, each span has
   `size`, `flags` (bold bit = 16), `bbox`.
3. **Body font size** = the most common size in the document, weighted by number of
   characters (`Counter`). A heading is
   `size >= body * 1.15`, at most 120 chars, at most 3 lines, and not a caption:

   ```python
   def _is_heading(self, tb: _TextBlock, body_size: float) -> bool:
       return (
           tb.size >= body_size * self.heading_ratio
           and len(tb.text) <= self.max_heading_chars
           and tb.n_lines <= 3
           and not _CAPTION.match(tb.text)
       )
   ```
4. **Heading levels**: all heading font sizes in the doc are sorted, biggest = level 0.
   A stack produces the `section_path`:

   ```python
   while stack and stack[-1][0] >= level:
       stack.pop()
   stack.append((level, " ".join(payload.text.split())))
   ```
   For example, "2 Results" (level 1) pops "1 Introduction" (level 1) but keeps the
   title (level 0), so the section path becomes `["Northwind Survey Report", "2 Results"]`.
5. **Tables**: `page.find_tables()` detects ruled tables; `t.to_markdown()` gives
   Markdown. Text blocks that lie >50% inside a table bbox (`_overlap_ratio`) are
   replaced by that table (so table text isn't indexed twice). False positives
   with < 2 rows or < 2 columns are dropped. `find_tables` is wrapped in
   `try/except`, so a broken page loses its tables but the document still parses.
6. **Noise filter**: blocks in the top/bottom 6% of the page shorter than 80 chars
   (running headers/footers) and bare page numbers are dropped.
7. **Captions** (`^(table|figure|fig.|chart|exhibit) N`): a "Table ..." caption
   within 80pt of a table becomes that table's `caption`. Other captions become an
   `image` Block. We can't see inside the chart, but its title makes the page
   findable.

Run it on the fixture:

```powershell
uv run python scripts/make_tiny_pdf.py
uv run python -c "from pathlib import Path; from docrag.ingest.pipeline import ingest_pdf; [print(c.id, c.type, c.pages) for c in ingest_pdf(Path('fixtures/tiny.pdf'))]"
```

```
tiny:p1:c0 text [1]
tiny:p2:c0 text [2]
tiny:p2:c1 table [2]
tiny:p3:c0 text [3]
```

### 3. The chunker (`chunker.py`)

Rules (from the module docstring):

1. Text blocks of the **same section** are packed together up to `max_words=200`.
   A section change always starts a new chunk.
2. A table is **one chunk** (caption + Markdown). Only tables larger than
   `max_table_words=350` are split **by rows**, and every piece repeats caption +
   header row (`_split_table`).
3. Figure captions are their own chunk.
4. Pages pass through: `page_start = min(pages)`, `page_end = max(pages)`. A
   paragraph continuing on the next page is *not* split at the page break.
5. `text` stays raw. Only `embed_text` gets the optional heading prefix:

```python
prefix = " > ".join(section)
embed_text = f"{prefix}\n{text}" if opts.prefix_headings and prefix else text
```

Why keep `text` raw? It's what the user sees and what Verify matches quotes
against. If the prefix were in `text`, an LLM quoting the heading would "verify"
against text that isn't really in the document.

### 4. Stable chunk IDs (`shared/ids.py`)

```python
def make_chunk_id(doc_id: str, page: int, index: int) -> str:
    """Build a stable chunk ID like 'doc3:p2:c1'."""
```

`index` restarts on every page (`next_index[f"{doc_id}:{page_start}"]`). Same PDF +
same parser/chunker version → same IDs on every run. The doc_id regex
`[A-Za-z0-9._-]+` forbids `:`, `[`, `]` and spaces, so an ID can be embedded in a prompt as
`[doc:p2:c1]` and parsed back with `parse_chunk_id`.

Qdrant point IDs must be ints or UUIDs, so:

```python
def chunk_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))
```

uuid5 is deterministic (a hash of namespace + name), so re-ingesting a chunk
overwrites the same point instead of creating a duplicate. Upserts are idempotent.

### 5. Parse cache and parallel indexing (`pipeline.py`, `indexer.py`)

`parse_pdf` looks up `parse_cache_key(bytes, parser, doc_id)` in the cache first.
`index_documents`:

1. skips docs already in the collection (`store.count(doc_id) > 0`). That's only
   safe because the collection name carries a fingerprint of the parser, chunker,
   embedder and OCR versions (`wiring.index_components`): bump `CHUNKER_VERSION`
   and you get a new, empty collection instead of silently reusing old chunks,
2. parses uncached PDFs in a `ProcessPoolExecutor` (4 workers),
3. writes the results to the cache (in the main process only, so there's one SQLite writer),
4. chunks → embeds → upserts one document at a time (bounded memory).

```python
def _parse_worker(args: tuple[Path, str]) -> tuple[str, list[Block]]:
    path, doc_id = args
    return doc_id, parse_pdf(path, doc_id, get_parser(), cache=None)
```

Measured: most MMLongBench PDFs take ~5 s, the 130-page 3M 10-K about 115 s,
almost all of it in `find_tables()`.

### 6. Page images (`pages.py`)

`render_page_png(path, page)` renders on demand (~50 ms) instead of pre-rendering
~6000 PNGs. `safe_pdf_path` refuses path traversal (`../../secrets`): the doc_id
is validated, the path is `resolve()`d, then `is_relative_to(base)` is checked.

## Why this design

> - **PyMuPDF instead of Docling:** ~1 s per 50 pages on CPU, no PyTorch, no model
>   downloads, Windows wheels. Docling has better layout and table quality but is
>   much heavier and slower. Behind the `Parser` Protocol, a `DoclingParser` is one class away.
>   Trade-off to mention: PyMuPDF is AGPL-3.0.
> - **Typed Block as the parser contract** (borrowed from RAG-Anything): the parser
>   decides page and section once; nobody reconstructs them later from text.
> - **Table = one chunk:** splitting a table mid-row destroys the header↔value link
>   that table questions depend on.
> - **Processes for parsing:** CPU-bound pure Python → the GIL blocks threads.

## Python concepts

### Closures and `nonlocal`

```python
def counter():
    count = 0

    def inc():
        nonlocal count  # without this, `count += 1` would create a NEW local
        count += 1
        return count

    return inc


c = counter()
c()
c()  # 2
```

In `chunk_blocks`, `emit()` only *mutates* the `chunks` list (no `nonlocal` needed),
but `flush()` *rebinds* `pending = _Pending()`, so it needs `nonlocal pending`. JS
closures capture variables by reference and rebinding just works. Java lambdas
can't rebind captured locals at all (they must be effectively final). Rust closures
need `move` or `&mut` borrows.

### The mutable default argument pitfall

```python
def add(item, bucket=[]):  # BUG: the list is created ONCE, at def time
    bucket.append(item)
    return bucket


add(1)
add(2)  # [1, 2]  (!)


def add(item, bucket=None):  # correct idiom, used in chunk_blocks(options=None)
    bucket = [] if bucket is None else bucket
```

ruff's `B006` rule catches this. TS/Java/Rust evaluate defaults per call, so this
trap is Python-specific.

### Context managers (`with`)

`with pymupdf.open(...) as doc:` ≈ Java try-with-resources, C# `using`, Rust `Drop`.
`__exit__` runs even on exceptions.

### Comprehensions and `enumerate`

```python
sizes = {tb.size for page in pages for tb in self._iter_text_blocks(page)}  # set
levels = {size: lvl for lvl, size in enumerate(sorted(sizes, reverse=True))}  # dict
```

### The GIL, processes, and the `__main__` guard

CPython runs Python bytecode in one thread at a time (Global Interpreter Lock).
Threads help when code *waits* (I/O) or calls C code that releases the GIL
(numpy, ONNX Runtime). They don't help for pure-Python CPU work like PyMuPDF's
table heuristics. `ProcessPoolExecutor` starts separate interpreters; arguments and
results are **pickled** across processes, so worker functions must be top-level
(not lambdas or closures). On Windows, child processes re-import the main module;
without `if __name__ == "__main__":` each child would start the program again.
(Python 3.13+ offers an optional free-threaded build without the GIL; we stay on 3.12.)

## Exercises

1. Change `heading_ratio` to `1.05` and re-run the tiny.pdf snippet. **Done when**
   you can explain what changed in `section_path` and why that's worse.
2. Set `ChunkerOptions(max_table_words=10)` and chunk tiny.pdf. **Done when** you
   see the table split into several chunks that each repeat the caption and header.
3. Write a test: chunk IDs of tiny.pdf are identical across two `ingest_pdf` calls.
   **Done when** it passes.
4. Call `safe_pdf_path(Path("data"), "../secret")`. **Done when** you know which of
   the two checks rejected it.
5. Time `index_documents` on 4 PDFs with `workers=1` vs `workers=4` (empty
   collection, empty parse cache). **Done when** you have both numbers.

## Interview questions

**Q: Why PyMuPDF and not Docling, Unstructured or an OCR pipeline?**
A: Speed and simplicity for v1: no ML models, ~1 s per 50 pages on CPU, robust on
Windows. Docling would give better tables and layout but costs PyTorch, downloads
and ~100x runtime. The Parser Protocol makes it a drop-in. I'd add it only if the
table slice of the benchmark shows parsing is the bottleneck.

**Q: How do you keep tables intact?**
A: Tables are detected per page, rendered to Markdown and emitted as one Block; the
chunker never merges or cuts them unless they exceed ~350 words. Then they're split by
rows, with caption and header repeated so every piece still answers "which value
belongs to which column".

**Q: How do you know which page a chunk is from?**
A: The parser sets `page` on every Block; the chunker takes min/max over the blocks it
packed into `page_start/page_end`. Nothing reconstructs pages from text offsets.
That's what makes page-level recall and citation-to-page possible.

**Q: Are chunk IDs stable? What breaks them?**
A: Yes, for the same PDF bytes and the same parser/chunker version they are
deterministic. Changing chunk size or parser logic changes IDs, which is why the version
goes into the cache key and you re-index into a fresh collection.

**Q: What's the heading prefix ablation?**
A: `embed_text` optionally starts with the heading chain ("Report > 2 Results"), so
short chunks carry their topic into the embedding. `text` stays raw. Collections
are named `...__pfx` or `...__raw`, and the results table compares both.
(fill in from results/)

**Q: Known weaknesses of the parser?**
A: Heading detection is a font-size heuristic, so catalogs with big marketing text
get noisy section paths. Two-column layouts rely on PDF stream order. Charts drawn as
vectors are seen only through their text labels. Scanned PDFs without a text layer
produce no chunks (logged, no OCR in v1).

**Q: Why processes, not threads or asyncio, for parsing?**
A: Parsing is CPU-bound Python. The GIL serializes threads and asyncio only helps
while waiting on I/O. Processes run in parallel; the cost is pickling the Blocks back,
which is small next to the parse time.
