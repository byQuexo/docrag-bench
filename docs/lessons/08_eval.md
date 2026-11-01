# Lesson 08: Evaluation (`src/docrag/eval/`, `configs/`)

**Goal:** understand how the benchmark turns a YAML config into a reproducible row in
the results table: dataset quirks, slices, page-level metrics, citation precision with
Verify off/on, the judge, and the output files. Milestone M5: "metrics.csv with all
core axes; a second run costs 0 EUR".

The plan's core question: *Which embedding model and which retrieval variant wins,
and does Verify make answers more honest?*

## What happens step by step

```
configs/x.yaml ─▶ ExperimentConfig ─▶ select questions (split, limits)
   ─▶ index_documents (only missing docs) ─▶ for each question:
         retrieval-only:  retriever.retrieve
         with llm:        AnswerService.answer (bounded_gather, limit 2)  ─▶ judge (limit 4)
   ─▶ question_row(...) per question ─▶ aggregate per slice ─▶ results/<stamp>_<name>_<hash>/
```

### 1. Dataset quirks (`eval/dataset.py`)

MMLongBench-Doc (Hugging Face `yubo2333/MMLongBench-Doc`): 1091 questions over 135 documents (134 PDFs downloadable)
(134 on disk), each question is about **one** document.

- `evidence_pages` and `evidence_sources` are **strings** like `"[19, 20]"` and
  `"['Table']"`. They're parsed with `ast.literal_eval`, never `eval`:

```python
def parse_list(raw: str) -> list[str | int]:
    value = ast.literal_eval(raw) if raw and raw.strip() else []
```

  `literal_eval` accepts only literals; `eval` would execute arbitrary code from the data.
- Pages are **1-based**. This was checked against a real table question: page index 18 in
  PyMuPDF = gold page 19.
- ~22% are `"Not answerable"` with empty evidence.
- `doc_id` has a `.pdf` suffix in the parquet; we strip it.

**Slices** from `evidence_sources`:

```python
_SOURCE_TO_SLICE = {
    "Pure-text (Plain-text)": "text",
    "Generalized-text (Layout)": "text",
    "Table": "table",
    "Chart": "image",
    "Figure": "image",
}
```

A question can be in several slices; unanswerable questions are only `"unanswerable"`.
Slices show *where* the system fails. Expect the `image` slice to be weak for a
text-only retriever; that's the argument for a ColPali-style visual retriever later.

`qid` = first 12 hex chars of `sha1(doc_id + question)`, stable across runs. The split is
by document (lesson 04). `select()` keeps only docs whose PDF exists, applies `split`,
`max_docs`, `max_questions`.

### 2. Page-level retrieval metrics (`eval/metrics.py`)

Ground truth is pages, retrieval returns chunks: map chunks to pages and deduplicate,
keeping the first occurrence:

```python
def ranked_pages(hits: list[Hit]) -> list[int]:
    seen: dict[int, None] = {}
    for hit in hits:
        for page in hit.chunk.pages:
            seen.setdefault(page, None)
    return list(seen)
```

- **Recall@k** = |gold ∩ top-k pages| / |gold|
- **MRR@10** = 1 / rank of the first gold page (0 if none in top 10)
- **nDCG@10**, binary relevance, implemented directly:

```python
dcg = sum(1.0 / math.log2(rank + 1) for rank, p in enumerate(pages[:k], 1) if p in gold)
idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gold), k) + 1))
return dcg / idcg
```

  The gotcha: the **ideal** DCG must count *all* gold pages (up to k), including ones we
  never retrieved. Otherwise retrieving 1 of 3 gold pages at rank 1 scores a "perfect" 1.0.

  The plan suggested `sklearn.metrics.ndcg_score`. sklearn only sees the items you pass
  and computes the ideal ranking from them. The first version appended the unretrieved gold
  pages with a low score, but when fewer than k pages were retrieved those pages landed
  *inside* sklearn's top-k and counted as hits. A unit test caught it. The formula is
  four lines, so it's implemented directly and sklearn remains a cross-check in
  `tests/unit/test_metrics.py`. That's a good interview story: "a test caught a metric bug".

Unanswerable questions have no gold pages → `NaN` → excluded from retrieval means.

### 3. Answer-level metrics

From `question_row`:

- `cit_precision_verify_off`: share of **all** LLM citations that point at a gold page.
- `cit_precision_verify_on`: the same, but only over citations that **passed Verify**.
- `status`, `abstained`, `n_citations`, `n_verified`.
- `correct`, `faithful` from the judge. `faithful` is NaN (= left out of the mean)
  unless the judge MODEL read the answer (`judged_by_llm`: answerable and not
  abstained). The rules only put a placeholder there.
- `failed`: the answer call raised (provider error, invalid JSON after the retry).
  Its mean per slice is the failure rate, and a failed answer counts as
  `correct = False`. Leaving it empty would silently drop it from the mean.

Verify on/off comes from the same answers, so there's no extra LLM call and the only
difference is Verify itself.

### 4. The judge (`eval/judge.py`): rules first, LLM second

```python
abstained = result.status == "abstained" or is_abstention(result.answer)
if not q.answerable:
    return JudgeVerdict(
        reasoning="rule: unanswerable", correct=abstained, faithful=True
    )
if abstained:
    return JudgeVerdict(reasoning="rule: abstained", correct=False, faithful=True)
```

Only the remaining cases go to the LLM with `prompts/judge_v1.txt` and a Pydantic schema:

```python
class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str  # first: the model justifies before committing
    correct: bool
    faithful: bool
```

Field order matters: with structured output the model generates fields in schema
order, so the reasoning comes before the booleans.

### 5. Outputs (`eval/runner.py`)

`results/<UTC stamp>_<name>_<config_hash>/`:

| file | content |
|---|---|
| `config.yaml` | the full config + hash: *what* was measured |
| `per_question.csv` | one row per question incl. `top_pages`, `gold_pages`, `answer`: for error analysis |
| `metrics.csv` | one row per slice (`all, text, table, image, unanswerable`), means ignoring NaN |
| `run.json` | wall time, `code` (git commit + dirty flag), `index` (collection + parser/chunker/embedder/OCR versions), p50/p95 per timer (`embed`, `search`, `rerank`, `llm`, ...; cache hits separately as `llm_cached` etc.), LLM calls, cache hits, tokens, cost, validation retries/failures |

`docrag summary` prints a Markdown table of the newest run per config (slice `all`) for
the README.

### 6. Reproducibility: second run = 0 EUR

- The config hash identifies the measurement.
- Parse, embed and LLM caches are content-addressed (lesson 01). A rerun hits the cache
  everywhere → `run.json` shows `cache_hits` = all calls, `cost_eur: 0`.
- Collections are reused: `index_documents` skips docs already indexed.

### 7. Running the experiment matrix

```powershell
docker compose up -d qdrant
uv run docrag calibrate --config configs/baseline.yaml       # dev split -> threshold
# put the threshold into the configs, then:
uv run docrag eval configs/retrieval_dense.yaml configs/retrieval_bm25.yaml `
                   configs/retrieval_hybrid.yaml configs/baseline.yaml `
                   configs/embed_modernbert.yaml configs/no_prefix.yaml
uv run docrag summary
```

Axes covered by the configs:

| axis | configs |
|---|---|
| retrieval variant | `retrieval_dense`, `retrieval_bm25`, `retrieval_hybrid`, `baseline` (hybrid_rerank) |
| embedding model (384 vs 768 dim) | `baseline` (gte-small) vs `embed_modernbert` |
| heading prefix | `baseline` vs `no_prefix` |
| Verify off/on | every run with `llm` set (two columns) |

For the final numbers set `dataset.split: test` and a larger `max_docs`.

### 8. Case study: the abstention experiment

The first answer runs abstained on ~43% of answerable questions. How it was
investigated is a template for any "why is metric X bad?" question:

1. **Measure the cause instead of guessing.** Two causes were possible: the rerank
   gate (abstain before the LLM) or the LLM itself. `per_question.csv` didn't say
   which, so the runner now writes `abstained_before_llm` and `top_score`. Result:
   the gate stops only ~2–3% of answerable questions. The LLM was the cause.
2. **Choose on dev, report on test.** Four variants (prompt `answer_v1`/`answer_v2`
   × `context_k` 5/10) ran on the dev split (`configs/dev_vlm_answer_*.yaml`). Only
   the winner ran on test. Comparing four variants on test and reporting the best
   would be tuning on the test set, the same mistake the threshold calibration avoids.
3. **Test the difference, paired.** Both variants answer the *same* questions, so
   the right test is McNemar's: only the questions where they disagree count.
   `answer_v2` + 10 chunks vs. the baseline on dev: 19 gained, 6 lost, p = 0.015.
   On test: 49 gained, 25 lost, p = 0.007.
4. **Split the result by slice.** The overall +3.2 points hides a trade-off:
   answerable questions +7.2 (48 gained, 7 lost), unanswerable −10.1 (1 gained,
   18 lost). Reading the 18 lost answers showed two patterns: the model says the
   premise is false *in prose* and adds a related fact (the exact-string abstention
   check doesn't see it), or it answers a near-miss (2013 data for a 2020 question).
   That diagnosis became the next experiment (`answer_v3`), again on dev first.

Along the way the judge got fixed too: `judge_v1` saw only the short quotes, so
answers that went beyond a 5-word quote looked unfaithful (0.46–0.50). `judge_v2`
sees the full cited passages (0.83–0.89). Changing the judge changes the measuring
instrument, so every row in the results table was re-judged with it.

## Why this design

> - **Page level, not chunk level:** the ground truth is pages, and it keeps different
>   chunkers comparable.
> - **Slices:** a single average hides that charts fail while text works.
> - **Rules before the LLM judge:** cheaper, deterministic, and unanswerable questions have
>   an exact definition of correct.
> - **Same pipeline as the API:** numbers describe the served system.

## Python concepts

### polars

```python
import polars as pl

df = pl.DataFrame([{"slice": "text", "r": 0.5}, {"slice": "table", "r": float("nan")}])
df.filter(pl.col("slice").str.contains("text"))
df["r"].cast(pl.Float64).fill_nan(None).mean()  # NaN -> null, mean skips nulls
```

polars is a DataFrame library like pandas, but it's written in Rust, has a lazy expression API,
and keeps types stricter. Think "SQL on in-memory tables". `pl.read_parquet("data/*.parquet")`
reads the dataset.

### Dict merge `|=`

`row |= {"correct": True}` ≈ `Object.assign(row, {...})` / `{...row, ...}`.

### `NaN` vs `None`

`float("nan")` is "not applicable"; `NaN != NaN` is True, so test with `math.isnan`.
`None` is Python's null.

### `datetime.now(UTC)`

Always use timezone-aware timestamps: `datetime.now()` without a tz is local and
ambiguous.

## Exercises

1. Hand-compute recall@1/@5, MRR@10 and nDCG@10 for retrieved pages `[7, 3, 9]` and gold
   `{3, 12}`, then call the functions. **Done when** they match.
2. In `ndcg_at_k`, change `idcg` to count only the gold pages that were *retrieved*, then
   rerun exercise 1. **Done when** you can explain the inflated number. Revert.
3. Run `configs/retrieval_bm25.yaml` with `max_docs: 5`, open `per_question.csv` and find
   the three worst questions. **Done when** you've looked at their `top_pages` vs
   `gold_pages` and the PDF pages and written down why they failed.
4. Run the same config twice with an LLM. **Done when** the second `run.json` shows
   `calls: 0` and only cache hits.
5. Add a `recall@20` column (needs `top_k: 20`). **Done when** it appears in `metrics.csv`.

## Interview questions

**Q: How do you evaluate retrieval when the ground truth is pages?**
A: Chunks are mapped to their pages, deduplicated in rank order, and then Recall@k, MRR@10 and
nDCG@10 are computed on that page ranking. Unanswerable questions have no gold pages and are
excluded from retrieval metrics.

**Q: Any pitfalls with nDCG?**
A: The ideal DCG must include every gold page, retrieved or not; otherwise partial retrieval
can score a perfect 1.0. sklearn's `ndcg_score` only knows the items you pass, and my
first workaround with it broke when fewer than k pages were retrieved. A unit test caught that, so
nDCG is now four lines of the formula, tested against hand calculations and cross-checked with sklearn.

**Q: How do you measure whether Verify makes answers more honest?**
A: Citation precision on all raw citations vs only verified ones, from the same answers,
plus the share of unsupported answers. (fill in from results/)

**Q: How reliable is your LLM judge?**
A: Rules handle the unambiguous cases (unanswerable, abstained); the judge only grades
correctness and faithfulness with a strict schema, reasoning first. A single judge has
bias, so it gets checked against a human: `docrag label-sample <run>` writes a *blind*
CSV of 50 judged answers (question, gold, answer, full cited passages, no verdicts),
I label them one by one in the terminal (`docrag label <csv>`, y/n per answer,
resumable), and `docrag judge-agreement <run> <csv>` reports raw agreement and Cohen's
κ (agreement corrected for chance) for `correct` and `faithful` (`eval/agreement.py`).
A panel of judges from different model families would be the next step.

**Q: How is a run reproducible?**
A: The config is saved with its hash; parse, embed and LLM results are cached by content and
version; the split is a deterministic hash. Rerunning the same config hits the cache: same
numbers, 0 EUR.

**Q: Why slices?**
A: The average hides failure modes. MMLongBench has many chart/figure questions that a
text-only retriever can't see, and the image slice makes that explicit and motivates the
visual retriever as future work. (fill in from results/)

**Q: What was the most surprising result?**
A: That two plausible-looking numbers were measurement artifacts. Faithfulness first
"dropped with OCR" (0.84 → 0.78), but only because abstentions counted as faithful
and OCR produced fewer of them. Then the corrected number (0.46) was low because the
judge saw only 5-word quotes; with the full passages it's 0.83–0.89. And the high
abstention rate wasn't the rerank threshold (as I assumed) but the prompt: "Do not
guess" made the model give up on table questions it could answer. Lesson: check the
instrument before you tune the system.

**Q: How did you decide that a prompt change really helped?**
A: Variants compared on dev only, winner run once on test, paired McNemar test on
the questions where the two variants disagree (test: 49 vs 25, p = 0.007), and the
result split by slice, which showed the cost: unanswerable questions got worse
(18 vs 1). A single "correct" number would have hidden that.
