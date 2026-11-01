Runs from the first full eval (2026-11-01) that were superseded:
- duplicates from two eval chains running at once
- rerank configs where the threshold cut hits (recall bug) and the answer run with HTTP 402 credit errors (no max_tokens)
Kept for honesty and for lesson 12. Not included in docrag summary.
- `corpus_vlm` (first try): scored with a metric that compared page numbers without the document, so in "all documents" search a page 5 of another PDF counted as a gold page 5 (recall@10 reported 0.658, too high). Fixed in eval/metrics.py (`doc_id`), re-run.
- `answer_model_gemini38flash` and `answer_model_gpt54mini` (first try, 2026-11-01): the OpenRouter key hit its monthly limit mid-run (HTTP 403), so 576 and 714 of 735 answers failed. Not results; to be re-run.
- `answer_model_qwen38flash` (first full run): 4 of its 45 failed answers were provider rate-limit errors (HTTP 429), not model errors. Re-run from the cache: only those questions were asked again. Its latency (p50 10.1 s, p95 68 s) is the one reported, since the re-run timed only 16 calls.
