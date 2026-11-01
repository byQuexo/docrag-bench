Runs from the first full eval (2026-11-01) that were superseded:
- duplicates from two eval chains running at once
- rerank configs where the threshold cut hits (recall bug) and the answer run with HTTP 402 credit errors (no max_tokens)
Kept for honesty and for lesson 12. Not included in docrag summary.
- `corpus_vlm` (first try): scored with a metric that compared page numbers without the document, so in "all documents" search a page 5 of another PDF counted as a gold page 5 (recall@10 reported 0.658, too high). Fixed in eval/metrics.py (`doc_id`), re-run.
