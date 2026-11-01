"""Cross-encoder reranking + a calibrated score threshold.

Bi-encoder (the embedder) vs cross-encoder (this):
* Bi-encoder embeds query and chunk SEPARATELY -> vectors can be precomputed and
  searched in milliseconds over millions of chunks, but the model never sees
  query and chunk together.
* Cross-encoder reads "[query] [SEP] [chunk]" as ONE input and outputs a relevance
  score. Much more accurate, but one forward pass per (query, chunk) pair, so we
  only run it on the top ~50 candidates from hybrid search.

The threshold turns "ranked list" into "relevant or not". If nothing passes, the
pipeline abstains instead of letting the LLM answer from irrelevant context.
That is how "Not answerable" questions can be answered correctly.
"""

from pathlib import Path
from typing import Protocol

import numpy as np

from docrag.shared import tracing
from docrag.shared.schema import Hit
from docrag.shared.timing import timed


class Reranker(Protocol):
    name: str

    def score(
        self, query: str, texts: list[str]
    ) -> list[float]: ...  # 0..1, higher = better


class OnnxCrossEncoder:
    """gte-reranker-modernbert-base via ONNX Runtime."""

    REPO = "Alibaba-NLP/gte-reranker-modernbert-base"

    def __init__(self, cache_dir: Path, batch_size: int = 16):
        from docrag.embed.onnx import OnnxFiles, load_session

        self.name = "gte-reranker-modernbert-base"
        files = OnnxFiles(
            repo=self.REPO,
            revision="f7481e6055501a30fb19d090657df9ec1f79ab2c",
            cpu_file="onnx/model_int8.onnx",
            gpu_file="onnx/model_fp16.onnx",
        )
        self.session, self.tokenizer, _ = load_session(files, cache_dir)
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding()
        self.batch_size = batch_size
        self._input_names = {i.name for i in self.session.get_inputs()}

    def score(self, query: str, texts: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            # Pair encoding: the tokenizer builds "[CLS] query [SEP] text [SEP]".
            enc = self.tokenizer.encode_batch([(query, t) for t in batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            feeds = {
                "input_ids": ids,
                "attention_mask": np.array(
                    [e.attention_mask for e in enc], dtype=np.int64
                ),
            }
            if "token_type_ids" in self._input_names:
                feeds["token_type_ids"] = np.array(
                    [e.type_ids for e in enc], dtype=np.int64
                )
            logits = self.session.run(None, feeds)[0].reshape(-1)
            # The model outputs a raw logit; sigmoid maps it to 0..1. The scale
            # depends on the model, which is why the threshold must be calibrated.
            scores.extend((1.0 / (1.0 + np.exp(-logits))).tolist())
        return scores


class FakeReranker:
    """Word-overlap reranker for tests: share of query words found in the text."""

    name = "fake"

    def score(self, query: str, texts: list[str]) -> list[float]:
        from docrag.shared.text import tokenize

        q = set(tokenize(query))
        return [len(q & set(tokenize(t))) / max(len(q), 1) for t in texts]


def make_reranker(spec: str, models_dir: Path) -> Reranker:
    backend, _, _model = spec.partition(":")
    if backend == "onnx":
        return OnnxCrossEncoder(models_dir)
    if backend == "fake":
        return FakeReranker()
    raise ValueError(f"Unknown reranker {spec!r}")


@timed("rerank", kind=tracing.RERANKER)
def rerank(
    reranker: Reranker,
    query: str,
    hits: list[Hit],
    top_k: int,
    threshold: float | None = None,
) -> list[Hit]:
    """Rescore hits, sort by the new score, return the best ``top_k``.

    ``threshold`` optionally drops hits scoring below it. The pipeline does NOT
    use it (see AnswerService): cutting individual hits destroyed recall in the
    first full run (recall@10 0.55 -> 0.22). The threshold now only gates the
    whole answer: "is even the BEST chunk relevant enough to answer?"
    """
    if not hits:
        return []
    scores = reranker.score(query, [h.chunk.text for h in hits])
    rescored = [
        Hit(chunk=h.chunk, score=s, source="rerank")
        for h, s in zip(hits, scores, strict=True)
        if threshold is None or s >= threshold
    ]
    rescored.sort(key=lambda h: h.score, reverse=True)
    out = rescored[:top_k]
    tracing.annotate(
        {
            "reranker.model_name": reranker.name,
            "reranker.query": query,
            "reranker.top_k": top_k,
            "docrag.n_candidates": len(hits),
        }
    )
    tracing.annotate_documents("reranker.output_documents", tracing.hit_docs(out))
    return out


def calibrate_threshold(scores: list[float], labels: list[bool]) -> tuple[float, float]:
    """Pick the threshold that maximizes F1 of "score >= t means positive".

    Used per QUESTION (see eval/calibrate.py): score = the best rerank score of
    the question's candidates, label = "the system should answer" (answerable
    and a gold page is among the candidates). Run on the DEV split
    only, then report results on TEST - choosing a threshold on the same data you
    report on is overfitting ("tuning on the test set").
    Returns (threshold, f1).
    """
    if not scores or not any(labels):
        return 0.5, 0.0
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    # Sort descending: "threshold = i-th score" keeps exactly the first i+1 items.
    # cumsum then gives true positives for EVERY threshold in one O(n log n) pass
    # (the naive loop over thresholds would be O(n^2)).
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp = np.cumsum(y)
    kept = np.arange(1, len(s) + 1)
    precision = tp / kept
    recall = tp / y.sum()
    f1 = np.where(
        precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0
    )
    # Ties: only a position where the next score differs is a real threshold.
    valid = np.append(s[1:] != s[:-1], True)
    best = int(np.argmax(np.where(valid, f1, -1.0)))
    return float(s[best]), float(f1[best])
