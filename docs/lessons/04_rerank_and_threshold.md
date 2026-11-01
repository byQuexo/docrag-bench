# Lesson 04: Reranking and the calibrated threshold (`rerank/`, `eval/calibrate.py`)

**Goal:** understand why a second-stage cross-encoder improves precision, how its
scores become a yes/no "relevant" decision, and why the threshold must be chosen on
dev data only.

## What happens step by step

### 1. Bi-encoder vs cross-encoder

| | Bi-encoder (embedder) | Cross-encoder (reranker) |
|---|---|---|
| Input | query and chunk **separately** | `[CLS] query [SEP] chunk [SEP]` **together** |
| Precompute? | yes, chunk vectors stored in Qdrant | no, one forward pass per pair |
| Speed | ms over millions of chunks | ~50 pairs per query is the budget |
| Accuracy | good recall | better precision: attends across query and chunk tokens |

So: hybrid search fetches 50 candidates (high recall), the reranker reorders them
and cuts off the irrelevant ones (high precision).

### 2. `OnnxCrossEncoder.score()` (`rerank/reranker.py`)

```python
enc = self.tokenizer.encode_batch([(query, t) for t in batch])  # pair encoding
...
logits = self.session.run(None, feeds)[0].reshape(-1)
scores.extend((1.0 / (1.0 + np.exp(-logits))).tolist())  # sigmoid -> 0..1
```

Passing tuples to `encode_batch` makes the tokenizer build the pair input with separator
tokens. The model outputs one raw logit per pair; sigmoid maps it to 0..1. The
**scale depends on the model**, so the threshold can't be a universal constant.

### 3. `rerank()`: rescore, cut, sort

```python
scores = reranker.score(query, [h.chunk.text for h in hits])
rescored = [
    Hit(chunk=h.chunk, score=s, source="rerank")
    for h, s in zip(hits, scores, strict=True)
    if s >= threshold
]
rescored.sort(key=lambda h: h.score, reverse=True)
return rescored[:top_k]
```

Note: the reranker reads `chunk.text` (raw), not `embed_text`.

If **nothing** passes the threshold, `AnswerService` abstains without calling the LLM
(lesson 06). That's how "Not answerable" questions can be answered correctly, and
it saves money.

### 4. Calibrating the threshold (`calibrate_threshold`)

For every **dev** question: take the 50 hybrid candidates, score them, and label a candidate
`True` if any of its pages is a gold evidence page. Then pick the threshold that maximizes F1
of "score ≥ t ⇒ relevant".

The naive version tries every threshold and counts → O(n²). The implementation is
O(n log n):

```python
order = np.argsort(-s, kind="stable")  # sort descending
s, y = s[order], y[order]
tp = np.cumsum(y)  # true positives if we keep the first i+1
kept = np.arange(1, len(s) + 1)
precision = tp / kept
recall = tp / y.sum()
f1 = np.where(
    precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
)
valid = np.append(
    s[1:] != s[:-1], True
)  # with ties, only the last of a run is a real cut
best = int(np.argmax(np.where(valid, f1, -1.0)))
return float(s[best]), float(f1[best])
```

Key insight: if the scores are sorted descending, "threshold = i-th score" keeps
exactly the first i+1 items, so a cumulative sum gives TP for every threshold at once.

`docrag calibrate --config configs/baseline.yaml` prints e.g.
`rerank_threshold: 0.3127   # dev F1 = 0.41`, which you copy into the YAML.

### 5. Dev/test split by document (`eval/dataset.py`)

```python
def doc_split(doc_id: str, dev_share: float = 0.3) -> Literal["dev", "test"]:
    bucket = int(hashlib.sha256(doc_id.encode()).hexdigest(), 16) % 1000
    return "dev" if bucket < dev_share * 1000 else "test"
```

- **Deterministic:** the same doc always lands in the same split, on every machine
  (sha256, not Python's randomized `hash()`).
- **By document, not by question:** questions about the same PDF share vocabulary
  and layout. Splitting by question would leak test documents into tuning.
- Choose the threshold (and RRF k) on dev, **report on test**. Choosing on the data you
  report on is overfitting.

## Why this design

> - **Two-stage retrieval** is the standard way to get both recall (cheap, wide) and
>   precision (expensive, narrow).
> - **gte-reranker-modernbert-base, int8 ONNX:** Apache-2.0, same family as the
>   embedder, ~150 MB, runs on CPU. bge-reranker-v2-m3 is the stronger multilingual
>   alternative (relevant for the "German" axis later).
> - **A threshold, not just top-k:** top-k always returns k chunks, even for a question
>   the document can't answer. A threshold allows "nothing relevant", which is
>   what makes honest abstention possible.
> - **0.85 from the plan was only a start value:** the score scale depends on the
>   model, and calibration replaces guessing with a measured number.

## Python concepts

### `zip(..., strict=True)`

```python
list(zip([1, 2, 3], ["a", "b"]))  # [(1,'a'), (2,'b')] - silently drops 3!
list(zip([1, 2, 3], ["a", "b"], strict=True))  # ValueError
```

ruff `B905` requires `strict=` to be explicit. Rust's `zip` truncates silently too; in TS
you'd usually index manually.

### Sorting with `key=`

```python
hits.sort(key=lambda h: h.score, reverse=True)  # in place
top = sorted(hits, key=lambda h: (h.chunk.doc_id, -h.score))  # new list, tuple key
```

Tuple keys sort lexicographically: Python's equivalent of a `Comparator` chain.

### `np.where` as a vectorized if/else

`np.where(cond, a, b)` picks elementwise: a vectorized `cond ? a : b`.

## Exercises

1. `calibrate_threshold([0.9, 0.8, 0.3, 0.1], [True, False, True, False])`. Compute
   precision/recall/F1 by hand for each threshold first. **Done when** your hand
   calculation matches the result.
2. Write the naive O(n²) version as a test oracle and compare on 1000 random
   scores/labels. **Done when** both agree.
3. Run `docrag ask` with a question the document can't answer, at threshold 0.0 and
   at the calibrated value. **Done when** you've seen "abstained" with the calibrated
   threshold.
4. Print `doc_split()` for 10 doc_ids twice in two separate Python processes. **Done
   when** you've confirmed it's stable, then try the same with `hash(doc_id) % 2` and
   see it change between runs.

## Interview questions

**Q: Why rerank at all if you already have hybrid search?**
A: First-stage retrieval optimizes recall over many chunks with precomputed vectors;
the cross-encoder reads query and chunk together and is much better at ordering the top
50. It's slower per pair, so it only runs on the candidates.
(fill in from results/: hybrid vs hybrid_rerank)

**Q: How did you choose the threshold?**
A: On the dev split only: label each candidate by whether it's on a gold evidence page,
then pick the score that maximizes F1. The result is reported on the test split. 0.85
from the plan was a placeholder; the score scale depends on the model.

**Q: Why split by document?**
A: Questions on the same document are correlated. A question-level split would put
the same PDF in dev and test and inflate test numbers after tuning.

**Q: What does the threshold do for unanswerable questions?**
A: If no candidate passes, the system returns "Not answerable" without an LLM call.
MMLongBench has ~22% such questions, and there abstaining is the correct answer.

**Q: Downsides of the threshold?**
A: A single global threshold ignores per-question difficulty; a question whose evidence is a
chart may never score high on text. Too high → false abstentions on answerable questions;
too low → the LLM sees noise. The eval reports abstention rates on both answerable and
unanswerable slices to make that trade-off visible.

**Q: What's the latency cost?**
A: About one forward pass per candidate: 50 pairs at 512 tokens on CPU int8, roughly a second.
The `rerank` timer in `run.json` gives p50/p95. (fill in from results/) A GPU or fewer candidates
reduce it.
