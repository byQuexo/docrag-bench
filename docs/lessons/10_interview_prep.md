# Lesson 10: Interview prep — question bank

How to use: cover the answer, say yours out loud, then compare. Answers are 2-5
sentences and name the file to open if you're asked to show it. "(fill in from
results/)" means: quote your real number from `uv run docrag summary`.

**More questions with real numbers** are at the end of lesson 11 (GPU, secrets,
security, the bugs found at full scale) and lesson 12 (OCR: RapidOCR vs vision LLM).
The **headline numbers** to know by heart (test split, see README "Results"):
hybrid + rerank recall@10 0.55 → **0.78 with vision-LLM OCR**; answers correct
0.39 → **0.50**; chart questions 0.10 → **0.31**; total LLM spend ~**1 EUR**.

**Your 30-second pitch:** "docrag-bench is a document RAG system over real PDFs with
a benchmark attached. PDFs are parsed into typed blocks that keep page numbers,
headings and tables intact, indexed in Qdrant with dense embeddings plus BM25 fused by
RRF, and reranked by a cross-encoder with a calibrated threshold. Answers come from any LLM
through my own proxy as structured JSON with chunk-ID citations and verbatim quotes, which are
verified in code. Every design choice is an axis in the benchmark on MMLongBench-Doc,
so I can show with numbers which embedding model and retrieval variant win and
whether verification makes the citations more honest."

---

## A. "Why X and not Y?"

**1. Why PyMuPDF and not Docling?**
PyMuPDF parses ~50 pages per second on CPU without ML models or PyTorch and has Windows
wheels; Docling gives better layout and table structure but is much heavier and slower. The
parser sits behind a `Parser` Protocol (`ingest/parser.py`), so Docling is one class away, and the
rule is to add it only if the table slice shows parsing is the bottleneck. The trade-off is PyMuPDF's
AGPL license, which matters for closed-source products.

**2. Why not OCR / a vision-language model for parsing?**
MMLongBench PDFs are born-digital, so the text layer is exact and free. OCR adds errors
and cost. Charts are the real gap; the planned answer is a visual retriever (ColPali-style)
on page images, not OCR.

**3. Why Qdrant and not pgvector, Chroma, Pinecone or Elasticsearch?**
Qdrant does dense + sparse + RRF fusion + payload filters in one server-side query, has
an in-process `:memory:` mode for tests, and runs locally in Docker for free. pgvector would
need BM25 elsewhere and manual fusion; Chroma has no native sparse/RRF; Pinecone is
managed and paid; Elasticsearch/OpenSearch can do hybrid but are heavier to operate.

**4. Why ONNX Runtime and not sentence-transformers?**
The install is ~50 MB vs several GB of PyTorch, the same model files run on CPU, CUDA or DirectML, and
it's what you'd deploy. The cost is implementing tokenization, pooling and truncation
myself (`embed/onnx.py`, ~40 lines), which also made me understand what the models do.

**5. Why gte-small and gte-modernbert-base?**
Both are permissively licensed, ship ONNX weights on Hugging Face, and differ in size
(384 vs 768 dim, ~33M vs ~149M parameters). That makes them a clean "is the bigger model worth it?" axis.
(fill in from results/)

**6. Why hybrid search and not just dense?**
Dense retrieval matches meaning, BM25 matches exact tokens: numbers, names, codes like
"10-K" or "Table 3". Financial reports and papers need both. The benchmark compares dense,
BM25, hybrid and hybrid+rerank on identical questions. (fill in from results/)

**7. Why RRF and not a weighted sum of scores?**
Cosine and BM25 scores live on incompatible scales, so a weighted sum needs normalization and
per-dataset tuning. RRF uses only ranks and has one parameter. Its weakness, ignoring score
magnitude, is compensated by the cross-encoder that reorders afterwards.

**8. Why RRF k=60 when Qdrant's default is 2?**
60 is the value from the original RRF paper and weighs the two lists more evenly; k=2
strongly favors each list's top ranks. Since the default differs silently, I made it an
explicit config value (`rrf_k`) and report it. It can be tuned on dev.

**9. Why a cross-encoder reranker?**
It reads query and chunk together, so it's much more precise than comparing two
independent vectors. It's too slow for the whole corpus, so it runs on the top 50 hybrid
candidates. Its threshold also enables abstention. (fill in from results/)

**10. Why your own LLM proxy and not LiteLLM?**
I wanted to own and demonstrate the gateway logic: format conversion, message order,
caching, cost accounting and budget, and later tracing. With two adapters that's small and fully tested.
In a product with many providers I'd use LiteLLM behind the same `Provider` interface.

**11. Why Pydantic structured output and not regex JSON repair?**
`response_format` / `output_config.format` constrain generation to the schema, Pydantic
validates, and one retry shows the model its validation error. Failures are counted.
Regex repair silently guesses intent and hides model problems.

**12. Why 200-word chunks?**
~260 tokens fit gte-small's 512-token window with room for the heading prefix, and are
specific enough that a hit points to a passage, not a whole section. Sections never mix,
and tables are kept whole. Chunk size would be the next ablation axis.

**13. Why evaluate at page level and not chunk level?**
The gold labels are pages, and different chunkers produce different chunks, so page level
is the only fair comparison. Chunks map to pages because every chunk carries `page_start/page_end`.

**14. Why split dev/test by document?**
Questions about the same PDF are correlated. A question-level split leaks test documents
into threshold tuning. The split is a deterministic sha256 of the doc_id (`eval/dataset.py`).

**15. Why SQLite for the cache and not Redis?**
Single-machine benchmark, zero operations, persistent, in the standard library. The cache is
an interface with namespaces; for a multi-replica API it would move to Redis.

**16. Why an in-memory thread store?**
It's enough for one process and for tests. The `ThreadStore` Protocol makes Redis a drop-in, and the
API itself is stateless.

**17. Why FastAPI?**
Type-hint-driven validation and OpenAPI docs, async support for LLM I/O, and it's the de-facto
standard for Python ML services. Flask would need extra libraries for validation and async.

**18. Why uv?**
One tool for Python versions, venvs, resolution and a real lockfile, and it's very fast in CI.

**19. Why MMLongBench-Doc as the dataset?**
Real PDFs (reports, papers, 10-Ks, brochures), page-level evidence labels, answerable and
unanswerable questions, and evidence types (text/table/chart/figure) that enable slices.
The plan said Kaggle, but the dataset is on Hugging Face, so the download script uses that.

**20. Why Verify in code instead of an LLM checking itself?**
It's exact, free, instant and deterministic. LLM self-checks are useful for *semantic*
support, which is a separate planned stage (LettuceDetect).

**21. Why put the heading chain only into `embed_text`?**
The embedding gets the topic, while the displayed and verified text stays exactly what's in
the document. Otherwise a quote of the heading would "verify" against text that isn't really
in the chunk. It's an ablation axis (`prefix_headings`). (fill in from results/)

**22. Why Protocols for some interfaces and an ABC for Provider?**
Protocols are structural: fakes and new backends don't need to inherit anything. Provider
is an ABC because it carries a shared template method (`chat()` = to_native → _call →
from_native).

## B. "What are known issues / limitations?"

**23. What are the main known issues?**
(1) Charts and figures are invisible except for captions and text labels, so the image slice is
weak. (2) The heading heuristic is noisy on marketing-style PDFs. (3) Follow-up questions
retrieve on the raw question, with no query rewriting from history. (4) One global rerank
threshold. (5) A single LLM judge, no agreement measurement yet. (6) Scanned PDFs without a text
layer produce no chunks.

**24. Limits of Verify?**
It proves the quote exists, not that it supports the claim; a correct quote with a wrong
conclusion passes. Very short quotes (e.g. one number) pass easily. Normalization can miss
legitimate matches (e.g. "%" vs "percent").

**25. Limits of BM25 here?**
English stopwords only, no stemming ("revenues" ≠ "revenue"), crc32 term-ID collisions
(rare), and a fixed `AVGDL`.

**26. Limits of the evaluation?**
The default config uses a subset of documents for CPU time; one judge model; the
judge is an LLM with its own bias; page-level labels can't tell *which* chunk on a page
was right.

**27. Performance bottlenecks?**
Table detection in PyMuPDF dominates parsing (~115 s for a 130-page 10-K), but it's
cached and parallelized. At query time, reranking 50 pairs on CPU (~1 s) and the LLM call dominate.
(fill in p50/p95 from results/run.json)

**28. Licensing caveats?**
PyMuPDF is AGPL; the models are Apache-2.0/MIT; the dataset license must be checked before
redistributing the PDFs (the repo doesn't include them).

## C. "What happens when…?"

**29. …Qdrant is down?**
The retriever's search call raises a connection error, which surfaces as HTTP 500. In
production I'd map it to 503 with Retry-After and add a readiness probe that checks
Qdrant; `/health` is liveness only.

**30. …the LLM returns invalid JSON twice?**
The first failure triggers one retry including the validation error. A second failure raises
`StructuredOutputError` and increments `validation_failures`. The API returns 502; the eval
records the failure, keeps the question's retrieval metrics and reports the count in
`run.json`.

**31. …the LLM invents a chunk ID?**
Verify check (a) fails: the citation is marked `verified=False, reason="id not in
context"`, its inline marker is removed, and the status drops to partially_supported or
unsupported. Only verified sources are returned by the API.

**32. …the LLM paraphrases the quote?**
Check (b) fails ("quote not found in chunk"), with the same handling. The prompt asks for
short verbatim quotes to make this rare.

**33. …the budget is exceeded?**
Before each uncached call the proxy compares accumulated cost against `budget_eur` and
raises `BudgetExceededError`, which becomes HTTP 402 or a counted eval failure. Cached calls still
work, because they're free.

**34. …a PDF is scanned, without a text layer?**
The parser finds no text blocks, the chunker emits no chunks, and the indexer logs a warning and
skips it. Questions on it retrieve nothing → abstain. The fix would be an OCR parser behind
the same Protocol.

**35. …two users write to the same thread concurrently?**
The store's lock prevents corruption, but both requests read the history before either
appends, so neither answer sees the other's turn. Strict ordering would need a
per-thread lock or optimistic versioning.

**36. …you change the parser?**
Bump `PyMuPdfParser.version`. The parse cache key changes, so all PDFs are re-parsed. Chunk IDs may
change, so re-index into a fresh collection (or delete the old one). Stale entries are never
read again.

**37. …you change the prompt?**
Create `answer_v2.txt` and set `prompt_version`. The prompt text is in the messages, so
the LLM cache key changes automatically and old answers aren't reused. The config hash
changes too, so results stay separated.

**38. …a table spans two pages?**
PyMuPDF detects it as two tables, one per page, so it becomes two chunks. The second one lacks the
caption and maybe the header. This is a known limitation; a fix is merging tables with identical
column structure across a page break in the parser.

**39. …someone asks about a chart?**
Only the caption ("Figure 3: …") and text labels drawn as text are indexed, so
retrieval may find the page but the LLM can't read the bars, and it should answer
"Not answerable" or give a partial answer. The image slice quantifies this; a visual
retriever plus a vision LLM is the planned fix.

**40. …1000 concurrent requests arrive?**
Reranking (CPU) and LLM rate limits become the bottleneck, and latency rises. I'd add
admission control (a semaphore, 429 when full), move embedding/reranking to a GPU
inference service, cache answers, scale API replicas behind a load balancer, and move threads/cache
to Redis.

**41. …you switch to an embedding model with a different dimension?**
`collection_name` includes model, dim and a fingerprint of parser/chunker/embedder/OCR
versions, so a new collection is created and filled;
the old one is untouched. Embedding cache keys include the model name. Both can be benchmarked
side by side.

**42. …the question has no keywords (only stopwords)?**
The BM25 query vector is empty; hybrid falls back to dense-only and `bm25` mode returns
nothing.

**43. …nothing passes the rerank threshold?**
`AnswerService` returns "Not answerable" with `abstained_before_llm=True`, without an LLM
call, which is correct for unanswerable questions and cheap.

**44. …a document is ingested twice?**
`index_documents` skips docs that already have points. Even if forced, point IDs are
uuid5 of the chunk ID, so upserts overwrite instead of duplicating.

**45. …a user requests `/pages/../../etc/passwd/1.png`?**
`safe_pdf_path` rejects the doc_id (regex) and would also reject any path that resolves
outside the PDF directory (`is_relative_to`) → HTTP 400.

**46. …the reasoning model wraps its answer in `<think>` tags?**
`strip_think` removes them (also when only the closing tag is present) before validation;
separate `reasoning_content` fields are dropped.

**47. …the provider returns 429?**
The OpenAI-compatible adapter retries 408/409/429/5xx with exponential backoff (1 s, 2 s),
and the Anthropic SDK retries itself. In the eval, `bounded_gather` limits concurrency to
avoid it in the first place.

**48. …you run the same benchmark twice?**
Parse, embed and LLM results come from the cache and existing collections are reused, so the second run
costs 0 EUR and produces the same numbers. `run.json` shows only cache hits.

## D. "How would you scale / productionize it?"

**49. What would you change for production?**
Redis for threads and cache; Qdrant as a cluster with snapshots; GPU inference service
(e.g. Triton or TEI) for embedder and reranker; trace sampling, content redaction and an OTel
Collector in front of the tracing backend (tracing itself exists, lesson 13); auth and rate
limiting on the API; streaming responses; readiness checks; structured logging.

**50. How would you monitor quality in production?**
Log the verify status distribution, rejected-citation rate, abstention rate and latency
per stage; sample traffic for offline judging; alert on shifts. Run the benchmark as a
regression gate in CI (Promptfoo is on the "later" list).

**51. How would you support many documents (corpus-wide search)?**
`scope: corpus` already exists. Add document-level routing first (metadata, a summary
index), then chunk search inside the top documents. Watch latency and recall@k on the
`corpus` config.

**52. How would you handle charts?**
A ColPali-style retriever over page images (late-interaction multi-vector embeddings in Qdrant),
fused with the text retriever, plus a vision-capable LLM for answering from page images.
Evaluate on the image slice.

**53. How would you add German?**
Multilingual embedder and reranker (e.g. bge-m3 / bge-reranker-v2-m3), a language-aware BM25 tokenizer
with German stopwords and stemming, and a German question set. "Language" is a planned axis.

**54. How would you make the judge trustworthy?**
A panel of judges from different model families, majority vote, and Cohen's κ for agreement
(`sklearn.metrics.cohen_kappa_score`); calibrate on a small human-labeled set.

**55. How would you reduce cost?**
The cache first, then batch APIs for the benchmark (often ~50% off), small/cheap models for
judging, low reasoning effort, early abstention when nothing is relevant, and fewer context
chunks if accuracy holds.

**56. How would you reduce latency?**
GPU reranking or fewer candidates (`prefetch_k`), smaller int8 models, streaming the
answer, prompt caching (the static prefix comes first), and parallel embedding of query + BM25.

## D2. Questions from things that actually happened while building it

**D2.1 Where does the time go per query?**
From the first smoke run's `run.json` (CPU, 21 questions): Qdrant search p50 ≈ 13 ms,
query embedding p50 ≈ 18 ms, **cross-encoder rerank p50 ≈ 8 s**. Search is basically free;
the reranker is >99% of retrieval latency because it runs a full transformer forward pass
for each of the 50 (query, chunk) pairs. That's measured, not guessed, thanks to the `@timed`
decorator - which is exactly why I added timing before tracing.

**D2.2 Which slice is weakest, and why?**
In the smoke run, charts/figures (the `image` slice): recall@10 ≈ 0.68 vs ≈ 0.88 for plain
text. The parser only keeps text and captions; what a bar chart *shows* is invisible to a
text embedder. The fix is a visual retriever (ColPali-style page embeddings) as another
retrieval branch fused with RRF - it's in Future work because now there's a number that
justifies it. (Re-check against the full run in results/.)

**D2.3 Did your tests ever catch a real bug?**
Yes: my first nDCG implementation used `sklearn.metrics.ndcg_score` and padded unretrieved
gold pages with a low score. When fewer than k pages were retrieved, those padded pages
fell *inside* sklearn's top-k and were counted as hits - `ndcg([5, 8], {5,6,7})` gave 0.91
instead of 0.47. A hand-calculated unit test caught it; I replaced it with the 4-line
formula and kept sklearn as a cross-check in the case where both definitions agree. The
lesson: metric code needs tests against hand calculations, because a wrong metric makes
every comparison wrong silently.

**D2.4 Why not just use OpenRouter for everything?**
I added OpenRouter as one more OpenAI-compatible provider - one key gives every model
family, which makes a multi-family judge panel a config change, and it reports the billed
cost per call (`Usage.cost_usd`), so the budget cap uses real numbers. But I kept the direct
Anthropic adapter: it's where the format conversion actually happens (top-level system,
`output_config`, content blocks, refusals). Trade-offs of OpenRouter: an extra network hop,
a small credit fee, data handling depends on the upstream host, and not every host supports
JSON-schema output - so requests set `provider.require_parameters: true` to route only to
hosts that support everything we send.

## E. "Tell me about the process / what you learned"

**57. What was the hardest part?**
Probably the data plumbing: keeping page numbers and table structure intact from
PDF to citation, and making the evaluation honest (page mapping, the nDCG gotcha, the split by document,
rules before the judge). (Personalize.)

**58. What surprised you in the results?**
(fill in from results/)

**59. What would you do differently?**
Start the benchmark harness even earlier and add a chunk-size axis; test table quality with
Docling on a few PDFs before committing to PyMuPDF; add query rewriting for threads.
(Personalize.)

**60. Where did you use AI tools, and how do you make sure you understand the code?**
Be honest: the project was developed in an agentic development workflow. Say concretely
how you reviewed and verified the code; the tests and the benchmark are the objective
checks. (Personalize.)

**61. How do you know your system works?**
Unit tests against fakes, one E2E against a real Qdrant in CI, mypy strict, and the
benchmark itself: retrieval recall per slice, answer correctness, citation precision with and
without Verify. (fill in from results/)

**62. What's the one rule you followed against over-engineering?**
"Every feature produces a number in the results table or answers an interview
question; everything else goes to Future Work." That's why there's no knowledge graph
(RAG-Anything's own ablation showed +3.4 pp for several LLM calls per chunk) and no plugin
architecture beyond two interfaces.

**63. Why not build a knowledge graph like RAG-Anything?**
Entity/relation extraction costs several LLM calls per chunk, and RAG-Anything's own
ablation shows only ~3.4 pp on DocBench. Hybrid + RRF + reranker is cheaper, measurable and
explainable. RAG-Anything could be added later as an optional comparison baseline.

**64. What's your differentiator vs. RAG-Anything?**
It returns answers as strings without structured citations. docrag-bench returns evidence
IDs with quotes, verifies them in code, and measures citation precision.

---

## Python rapid-fire (things interviewers like to check)

- **GIL?** One thread runs Python bytecode at a time; use processes for CPU-bound
  Python, threads/asyncio for I/O; C extensions (numpy, ONNX) release the GIL.
  → `ingest/indexer.py`, `answer/service.py`.
- **Mutable default argument?** Evaluated once at definition; use `None`.
  → `ingest/chunker.py`.
- **`is` vs `==`?** Identity vs equality; use `is` only for `None`.
- **Generator vs list?** Lazy vs materialized; `bounded_gather` takes an iterable of
  coroutines.
- **Decorator?** A function wrapping a function. → `shared/timing.py`.
- **Context manager?** `with` + `__enter__/__exit__` or `@contextmanager`.
  → `api/app.py` lifespan.
- **dataclass vs Pydantic?** Plain data vs validated data. → lesson 01.
- **`async` gotcha?** Calling a coroutine function doesn't run it; blocking calls freeze the
  loop. → `asyncio.to_thread`.
- **Why `ast.literal_eval`?** It safely parses literals; `eval` executes code. → `eval/dataset.py`.
