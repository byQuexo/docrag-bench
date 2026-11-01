"""Choose the rerank threshold on the DEV split (never on test).

For every dev question: take the hybrid candidates, score them with the reranker,
label each candidate "relevant" if it lies on a gold evidence page, then pick
the threshold with the best F1. Report results on test with that threshold.
"""

import logging

from docrag.eval.dataset import load_questions, select
from docrag.ingest.indexer import index_documents
from docrag.rerank.reranker import calibrate_threshold
from docrag.shared.cache import Cache
from docrag.shared.config import DatasetConfig, ExperimentConfig
from docrag.shared.settings import Settings
from docrag.wiring import build_ocr, build_retriever

log = logging.getLogger(__name__)


def calibrate(
    config: ExperimentConfig, settings: Settings, cache: Cache
) -> tuple[float, float]:
    dev_cfg = config.model_copy(
        update={
            "retrieval": "hybrid_rerank",
            "dataset": DatasetConfig(
                split="dev",
                max_docs=config.dataset.max_docs,
                max_questions=config.dataset.max_questions,
            ),
        }
    )
    questions = [
        q
        for q in select(
            load_questions(settings.qa_glob), dev_cfg.dataset, settings.pdf_dir
        )
        # unanswerable questions included on purpose: they are the negatives
    ]
    retriever = build_retriever(settings, dev_cfg, cache)
    doc_ids = sorted({q.doc_id for q in questions})
    index_documents(
        {d: settings.pdf_dir / f"{d}.pdf" for d in doc_ids},
        retriever.store,
        retriever.embeddings,
        prefix_headings=dev_cfg.prefix_headings,
        cache=cache,
        ocr=build_ocr(settings, dev_cfg, cache),
    )
    assert retriever.reranker is not None
    # One (score, label) pair per QUESTION, because the threshold makes a
    # per-question decision: answer or abstain.
    #   score = best rerank score among the question's candidates
    #   label = the system SHOULD answer: answerable AND a gold page is among the
    #           candidates (if retrieval missed the evidence, abstaining is right)
    # A first version labelled every (question, chunk) pair "on a gold page?" and
    # picked 0.874 - a good chunk filter, but as an abstain gate it made the
    # system refuse 92% of questions.
    scores: list[float] = []
    labels: list[bool] = []
    for q in questions:
        hits = retriever.candidates(q.question, [q.doc_id])
        if not hits:  # no text layer at all: the system abstains regardless
            continue
        gold = set(q.evidence_pages)
        best = max(retriever.reranker.score(q.question, [h.chunk.text for h in hits]))
        scores.append(best)
        labels.append(q.answerable and any(set(h.chunk.pages) & gold for h in hits))
    threshold, f1 = calibrate_threshold(scores, labels)
    log.info(
        "calibrated on %d dev questions (%d should be answered): "
        "threshold=%.4f (F1=%.3f)",
        len(scores),
        sum(labels),
        threshold,
        f1,
    )
    return threshold, f1
