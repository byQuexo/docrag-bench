# Lesson 03: Embeddings and hybrid search (`embed/`, `store/`)

**Goal:** understand how chunks become vectors, how Qdrant stores dense + BM25 in
one collection, and how Reciprocal Rank Fusion combines them. Milestone M2:
"Recall@10 on 20 questions is printed".

## What happens step by step

### 1. Backend vs service (`embed/base.py`)

```python
class Embedder(Protocol):
    name: str
    version: str
    dim: int

    def embed(self, texts: list[str]) -> Vectors: ...  # raw, shape (len(texts), dim)
```

A backend only turns texts into raw vectors. `EmbeddingService` does the rest
**once for all backends**:

1. look up each text in the cache (`stable_hash(name, version, text)`),
2. embed only the misses, in batches of `batch_size`,
3. L2-normalize,
4. write the new vectors to the cache as raw `float32` bytes.

```python
for start in range(0, len(missing), self.batch_size):
    idx = missing[start : start + self.batch_size]
    vecs = l2_normalize(self.embedder.embed([texts[i] for i in idx]))
    out[idx] = vecs
```

`make_embedder("onnx:gte-small", models_dir)` picks the backend by prefix
(`embed/registry.py`, a `match` statement). `fake:64` gives the deterministic
hashing-trick embedder used in tests.

### 2. ONNX embedding (`embed/onnx.py`)

```python
ONNX_MODELS = {
    "gte-modernbert-base": OnnxModelSpec(
        repo="Alibaba-NLP/gte-modernbert-base",
        onnx_file="onnx/model_int8.onnx",
        dim=768,
        pooling="cls",
    ),
    "gte-small": OnnxModelSpec(
        repo="thenlper/gte-small", onnx_file="onnx/model.onnx", dim=384, pooling="mean"
    ),
}
```

Per batch:

1. `tokenizer.encode_batch(texts)` (HF `tokenizers`, Rust under the hood),
   truncated to 512 tokens, padded to the longest in the batch.
2. Build numpy `int64` arrays `input_ids` and `attention_mask`. BERT-style models also
   want `token_type_ids`, ModernBERT doesn't, so we ask the session which inputs it has.
3. `session.run(None, feeds)[0]` → `last_hidden_state`, shape `(batch, tokens, hidden)`.
4. **Pooling** (from each model's `1_Pooling/config.json`):
   - CLS: `hidden[:, 0, :]` (the first token stands for the whole text).
   - Mean: average over real tokens only. Padding must not dilute the average:

```python
m = mask[:, :, None].astype(np.float32)
pooled = (hidden * m).sum(axis=1) / np.clip(m.sum(axis=1), 1e-9, None)
```

5. Back in the service: `l2_normalize` → every vector has length 1, so the dot product equals
   cosine similarity.

The two models are the "embedding model" axis of the plan: 768 dim vs 384 dim.

### 3. One collection per model (`store/qdrant.py`)

```python
def collection_name(embedder: str, dim: int, prefix_headings: bool) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "-", embedder)
    return f"chunks__{safe}__{dim}__{'pfx' if prefix_headings else 'raw'}"
```

Vectors from different models live in different spaces; mixing them in one collection is
meaningless. Naming makes accidental mixing impossible. Each point has:

- `dense`: cosine vector of `embed_text`,
- `bm25`: a sparse vector (next section),
- payload: the full Chunk JSON, so a hit can be displayed and cited without a second
  database.

A `KEYWORD` payload index on `doc_id` makes "search only in this document" a cheap
filter.

### 4. BM25 as sparse vectors (`store/bm25.py`)

**Correction to the plan:** Qdrant's full-text payload index only *filters* ("does it
contain these words: yes/no"); it doesn't *rank*. Ranked keyword search uses sparse vectors
with `Modifier.IDF`. The BM25 formula splits cleanly:

```
score(q,d) = Σ_t∈q  IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|d|/avgdl))
                    ^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                    Qdrant   we compute per chunk at index time
```

```python
for term, freq in tf.items():
    w = freq * (K1 + 1) / (freq + K1 * (1 - B + B * dl / avgdl))
    tid = term_id(term)  # crc32(term)
    weights[tid] = weights.get(tid, 0.0) + w
```

- IDF needs live corpus statistics, which change as docs are added, so Qdrant computes it
  at query time.
- The query side is weight 1.0 per unique term.
- `AVGDL = 90.0` is a fixed constant, so a chunk's vector never depends on which other
  documents were ingested in the same run.
- Term IDs are `crc32` hashes. Collisions are possible but rare (known, accepted; Qdrant's own
  BM25 does the same).

### 5. Search modes and RRF (`QdrantStore.search`)

- `dense`: `query_points(query=vector, using="dense")`
- `bm25`: `query_points(query=sparse, using="bm25")`
- `hybrid` / `hybrid_rerank`: two prefetch branches inside Qdrant, fused by RRF:

```python
res = self.client.query_points(
    self.collection,
    prefetch=[
        models.Prefetch(
            query=_dense(query_vector), using=DENSE, filter=flt, limit=prefetch_k
        ),
        models.Prefetch(query=sparse, using=SPARSE, filter=flt, limit=prefetch_k),
    ],
    query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
    limit=limit,
    with_payload=True,
)
```

RRF: `rrf(d) = Σ_branches 1 / (k + rank_branch(d))`. It only uses **ranks**, so
cosine (0..1) and BM25 (0..30+) scores never need calibrating against each other. `k`
damps the head of each list: with `k=60` (original paper) rank 1 vs 2 differ little;
Qdrant's default is `k=2`, which trusts the top ranks much more. We set it explicitly
(`rrf_k: 60` in the config) and report it.

If the query has no usable keywords (all stopwords), hybrid falls back to dense.

### 6. `store/retriever.py`

```python
def candidates(self, question, doc_ids=None):
    needs_vector = cfg.retrieval != "bm25"
    limit = cfg.prefetch_k if cfg.retrieval == "hybrid_rerank" else cfg.top_k
```

With rerank we fetch `prefetch_k=50` candidates for the reranker, otherwise `top_k`. The API
and the eval use this same object.

`scope: doc` in the config passes `doc_ids=[q.doc_id]`. MMLongBench is single-document
QA, so we search only that PDF. `scope: corpus` is the much harder variant.

## Why this design

> - **ONNX Runtime, not sentence-transformers/PyTorch:** ~50 MB vs several GB, the same
>   files run on CPU/CUDA/DirectML, and it's what you'd ship in a container. The cost:
>   we implement pooling and truncation ourselves (~40 lines).
> - **int8 ModernBERT:** about 4x smaller and 2-3x faster on CPU, tiny quality loss.
> - **Qdrant:** dense + sparse + RRF + filters in one engine, a server-side query plan, and an
>   embedded/`:memory:` mode for tests. pgvector would need BM25 elsewhere; Chroma
>   has no native sparse/RRF; Elasticsearch/OpenSearch can do it but are heavier to run.
> - **Hybrid:** dense finds paraphrases, BM25 finds exact names, numbers and codes
>   ("10-K", "Table 3", "1,500"). Documents with tables and numbers need both.

## Python concepts

### numpy shapes, broadcasting, fancy indexing

```python
import numpy as np

v = np.array([[3.0, 4.0], [1.0, 0.0]])  # shape (2, 2)
norms = np.linalg.norm(v, axis=1, keepdims=True)  # shape (2, 1): [[5.], [1.]]
v / norms  # broadcasting: each row / its norm
out = np.zeros((4, 2))
out[[0, 3]] = v  # fancy indexing: write rows 0 and 3
mask = np.array([[1, 1, 0]])[:, :, None]  # add a trailing axis: (1, 3) -> (1, 3, 1)
```

Think of an ndarray as a typed contiguous buffer with a shape (like Rust's
`Vec<f32>` plus dims). Operations run in C, so you avoid Python loops.

### `match` statement

```python
match backend:
    case "onnx":
        ...
    case "fake":
        ...
    case _:
        raise ValueError(...)
```

Similar to Rust `match` / Java 21 `switch`. It can also destructure:
`case {"type": "text", "text": t}:`.

### `str.partition`

`"ollama:qwen3:8b".partition(":")` → `("ollama", ":", "qwen3:8b")`, which splits on the
first separator only. Used for every `"backend:model"` spec.

## Exercises

1. Embed `["revenue in Europe", "European revenue", "delivery times"]` with
   `make_embedder("onnx:gte-small", get_settings().models_dir)` + `EmbeddingService`
   and print the 3x3 similarity matrix `v @ v.T`. **Done when** the first two are
   clearly closer than the third.
2. Change gte-small's pooling to `"cls"` and redo exercise 1. **Done when** you've seen
   the effect of wrong pooling (similarities get worse/flatter).
3. Print `doc_sparse_vector("revenue revenue revenue Europe")`. **Done when** you
   can explain why the weight of "revenue" is not 3x that of "europe".
4. Ingest tiny.pdf into `QdrantClient(":memory:")` with `FakeEmbedder` and compare the
   top hit for "Asia revenue 2024" in `dense`, `bm25` and `hybrid` mode. **Done when**
   you can explain differences in order.
5. Run `configs/retrieval_hybrid.yaml` with `rrf_k: 2` vs `60`. **Done when** you
   have both Recall@10 numbers.

## Interview questions

**Q: Why hybrid search and not just dense embeddings?**
A: Dense retrieval matches meaning but is weak on exact tokens: numbers, IDs, product
names, rare terms. BM25 is the opposite. On documents full of tables and figures both
matter; the benchmark compares dense / BM25 / hybrid / hybrid+rerank on the same
questions. (fill in from results/)

**Q: Why RRF and not a weighted sum of scores?**
A: The scores live on different scales and distributions (cosine vs unbounded BM25);
a weighted sum needs per-dataset normalization and tuning. RRF uses only ranks, has one
parameter, and is robust. The downside is that it ignores how confident a branch is. The
cross-encoder reranker after RRF fixes the final order anyway.

**Q: What's k in RRF and why did you set it?**
A: It dampens how much the top ranks dominate: 1/(k+rank). The paper uses 60, Qdrant
defaults to 2. That's a silent behavior difference, so I made it an explicit config
value and report it.

**Q: How do you implement BM25 in Qdrant?**
A: Sparse vectors with `Modifier.IDF`. I compute the TF-saturation and length-normalization part per
chunk at index time; Qdrant multiplies by IDF at query time from live stats. The full-text
index only filters, it doesn't rank.

**Q: Why one collection per embedding model?**
A: Vectors from different models aren't comparable and may differ in dimension. One
collection per (model, dim, prefix flag) makes mixing impossible and lets both models
be benchmarked side by side.

**Q: CLS vs mean pooling?**
A: It depends on how the model was trained. gte-modernbert-base uses CLS, gte-small
uses mean; I take it from each model's `1_Pooling/config.json`. Using the wrong pooling
silently degrades retrieval, with no error.

**Q: What happens if you swap to a model with a different dimension?**
A: `collection_name` changes (`chunks__onnx-new__1024__pfx__v<fingerprint>`), so a new
collection is created and filled; the old one is untouched. The fingerprint suffix hashes
parser, chunker, embedder and OCR versions, so even a same-dimension change (new pooling,
new chunker) gets a fresh index. The embed cache key includes the model name,
so nothing stale is reused.

**Q: Why normalize vectors?**
A: After L2 normalization the dot product equals cosine similarity, and every model is compared
on the same scale. It's done once in `EmbeddingService`, not in each backend.
