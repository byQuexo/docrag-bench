<p align="center">
  <a href="https://huggingface.co/spaces/Quexoo/docrag-bench"><img alt="docrag-bench: question answering over real PDFs with hybrid retrieval, reranking and verified citations, plus a benchmark that shows with numbers which choices actually help" src="docs/assets/readme/banner.svg" width="100%"></a>
</p>

<p align="center">
  <a href="https://github.com/byQuexo/docrag-bench/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/byQuexo/docrag-bench/ci.yml?style=for-the-badge&logo=githubactions&logoColor=white&label=CI"></a>
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-10b981?style=for-the-badge"></a>
  <a href="https://huggingface.co/spaces/Quexoo/docrag-bench"><img alt="Demo on Hugging Face" src="https://img.shields.io/badge/Demo-Hugging%20Face%20Space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"></a>
  <a href="docs/lessons/README.md"><img alt="13 interactive lessons" src="https://img.shields.io/badge/Lessons-13%20interactive-F37626?style=for-the-badge&logo=jupyter&logoColor=white"></a>
</p>

<p align="center">
  <img alt="0.544 correct [0.501, 0.590] with verified citations (citation precision 0.789) · scanned PDFs recall@10 0.000 → 0.776 → 0.834 (no OCR → RapidOCR → vision LLM) · +3.6 points correct, 0.508 → 0.544, McNemar p = 0.002, from the abstention experiment" src="docs/assets/readme/metrics.svg" width="100%">
  <br>
  <sub>MMLongBench-Doc test split: 735 questions over 93 PDFs, 23% unanswerable on purpose. 95% CIs bootstrapped over documents. Scanned PDFs: 152 answerable questions on 23 image-only PDFs. The 0.544 and the +3.6 points come from the <code>answer_v3</code> prompt, which was chosen with test results in view, from a run made with uncommitted code: read <a href="#how-far-to-trust-these-numbers">how far to trust these numbers</a> before quoting them.</sub>
</p>

docrag-bench answers questions over long PDFs (reports, papers, slide decks, scans) with page citations that are checked in code, and comes with the benchmark harness that measured which design choices actually help.

**Where it fits in my portfolio.** [Warden](https://github.com/byQuexo/warden) (an MCP security gateway in Rust) is the central project, [Steward](https://github.com/byQuexo/steward) the illustrative demo, and [agent-approval-gate](https://github.com/byQuexo/agent-approval-gate) its smaller predecessor. docrag-bench is the evidence of methodical evaluation work: a fixed split, paired tests, bootstrap intervals, a judge checked against human labels, and every number traceable to a file in [`results/`](results/).

## What it is

- **Problem.** RAG systems over long PDFs usually return an answer as a plain string, so you can't tell which sentence came from where, and claims like "reranking helps" or "the bigger model is better" are rarely measured with uncertainty.
- **What it gives you.** (1) Answers with structured evidence; every citation is checked in pure code (is the chunk ID one the model was given? is the quote verbatim in it?), and the system says "Not answerable" instead of guessing. (2) A benchmark harness: one YAML in [`configs/`](configs/) = one run; each run writes its config, per-question CSV, metrics and provenance to `results/`; intervals and paired tests are recomputed from those files with one command.
- **Scope: what it is not.** Not a production service (threads are in memory, no auth, single node). Not a general RAG framework. Not a leaderboard or state-of-the-art claim on MMLongBench-Doc. Not an independent held-out evaluation of the final prompt (see [limits](#how-far-to-trust-these-numbers)). Its "faithfulness" score is not a validated metric. English only.

## Try it

**Live demo: [huggingface.co/spaces/Quexoo/docrag-bench](https://huggingface.co/spaces/Quexoo/docrag-bench)**. Pick a sample PDF or upload your own (up to 60 pages), ask a question, and get the answer with verified page citations, the retrieved evidence and the cited page. Answers come from the open model Qwen3.8-27B on free ZeroGPU hardware: no API key, the GPU time comes from each visitor's daily Hugging Face quota. Expect a wait: the GPU is lent per question, and the first question about a new PDF also indexes it. The same demo runs locally ([demo/README.md](demo/README.md)).

**One concrete case from the benchmark** (question `aafd3a26978f` in [`per_question.csv`](results/20261101T162727_answer_haiku_vlm_v3_776cde4ae38c/per_question.csv) of the headline run):

| | |
|---|---|
| Document | `COSTCO_2021_10K` (Costco's 2021 10-K filing; gold evidence on pages 36 and 38) |
| Question | "What is total liabilities for costco in FY 2021?" (gold answer: 41190) |
| System answer | "Total liabilities for Costco in FY 2021 were $41,190 million [COSTCO_2021_10K:p38:c0]." |
| Checks | citation verified (status `supported`), cites a gold page (citation precision 1.0), judged correct |

The same run also has the other side: asked "What's the percentage of people who are democrats and voted in the last election compared to the entire population in 2024?" about `Independents-Report` (`8515c4089459`, gold answer "Not answerable"), it answers "Not answerable", which counts as correct.

## How it works

```json
{
  "answer": "Europe grew from 41.0 to 45.5 million EUR [report:p2:c1].",
  "evidence": [{ "id": "report:p2:c1", "quote": "Europe 41.0 45.5" }]
}
```

`Verify` checks every citation without an extra LLM call: the cited chunk must be one the LLM was given, and after normalization the quote must appear verbatim in it. Failed citations are removed; if none survive, the answer is labelled `unsupported` instead of being presented as sourced.

<img alt="Pipeline: PDFs → parse + OCR → chunk → embed into Qdrant (dense + BM25) → hybrid search + rerank → answer with verified citations; Eval runs the same retriever and answer service" src="docs/assets/readme/pipeline.svg" width="100%">

Three paths share one core: **Ingest** fills Qdrant, **Query** answers questions, and **Eval** calls exactly the same retriever and answer service as the API.

<details>
<summary>Detailed component diagram (Mermaid) and a screenshot</summary>

```mermaid
flowchart LR
  subgraph Ingest["Ingest (offline)"]
    PDF[PDFs] --> Parser["Parser (PyMuPDF)<br/>+ OCR for scanned pages"]
    Parser --> Chunker["Chunker<br/>table-aware, page + heading chain"]
    Chunker --> Embedder["Embedder (ONNX)"]
    Embedder --> Qdrant[("Qdrant<br/>dense + BM25 sparse")]
  end

  subgraph Query["Query (per request)"]
    API[FastAPI + threads] --> Hybrid["Hybrid search<br/>dense + BM25 → RRF"]
    Hybrid --> Qdrant
    Hybrid --> Rerank["Cross-encoder reranker<br/>+ calibrated abstain gate"]
    Rerank --> Proxy["LLMProxy<br/>cache, cost cap, structured output"]
    Proxy --> Verify["Verify<br/>ID in context? quote in chunk?"]
    Verify --> Answer[Answer + verified sources]
  end

  subgraph Eval["Eval (benchmark run)"]
    Set[MMLongBench-Doc] --> Runner[Runner<br/>same pipeline]
    Runner --> Metrics["Recall@k · MRR · nDCG<br/>citation precision · LLM judge"]
    Metrics --> Results[(results/)]
  end

  Runner -.-> Hybrid
```

![Chat with sources](docs/assets/chat.png)

*Chat with sources (a real question from the benchmark): the answer with `[p.X]` citations, the verified quotes, and the retrieved evidence after reranking with chunk type and score.*
</details>

## Results

Test split of [MMLongBench-Doc](https://huggingface.co/datasets/yubo2333/MMLongBench-Doc): 735 questions over 93 documents, split by document, search scoped to the question's document. Brackets are 95% CIs bootstrapped over documents (2,000 reps, seed 0). **Every number in this section is recomputed from the committed run files by `uv run python scripts/reproduce_stats.py`** (no data download, API key or index needed); the run folder behind each row is linked.

<p align="center">
  <img alt="Answer-model comparison, share correct with 95% CIs: Claude Haiku 5.5 0.544, Gemini 3.8 Flash 0.536, Qwen3.8-27B 0.521, Qwen3.8 Flash 0.520, GPT-5.4 mini 0.473, Qwen3.5-9B 0.434, Mistral Small 2603 0.423" src="docs/assets/readme/chart-answer-models.svg" width="49%">
  <img alt="Scanned PDFs recall@10: no OCR 0.000, RapidOCR 0.776, vision LLM 0.834; text-layer PDFs 0.752 / 0.753 / 0.755" src="docs/assets/readme/chart-scanned-ocr.svg" width="49%">
</p>

**Answers** (Claude Haiku 5.5 via OpenRouter, judged by DeepSeek V4.1 Flash with `judge_v2`):

| variant | correct | answerable | unanswerable | tables | charts | scanned PDFs | citation precision | faithful† | run |
|---|---|---|---|---|---|---|---|---|---|
| hybrid + rerank + LLM | 0.398 [0.362, 0.436] | 0.239 | 0.929 | 0.272 | 0.105 | 0.282 | 0.677 | 0.829 | [folder](results/20261101T144955_answer_haiku_8348ebba3c42/) |
| + OCR (RapidOCR) | 0.471 [0.433, 0.511] | 0.342 | 0.905 | 0.390 | 0.220 | 0.531 | 0.740 | 0.842 | [folder](results/20261101T145657_answer_haiku_ocr_e0374573a911/) |
| + OCR (vision LLM) | 0.508 [0.464, 0.556] | 0.397 | 0.881 | 0.390 | 0.309 | 0.653 | 0.762 | 0.885 | [folder](results/20261101T150405_answer_haiku_vlm_2156a0d5111d/) |
| + prompt `answer_v2`, 10 chunks | 0.540 [0.499, 0.585] | **0.469** | 0.780 | **0.566** | **0.357** | **0.667** | 0.764 | 0.851 | [folder](results/20261101T151804_answer_haiku_vlm_v2_7af24838e9f8/) |
| + prompt `answer_v3`, 10 chunks | **0.544** [0.501, 0.590] | 0.430 | **0.929** | 0.500 | 0.328 | 0.657 | **0.789** | 0.891 | [folder](results/20261101T162727_answer_haiku_vlm_v3_776cde4ae38c/) |

"Correct" counts abstaining on an unanswerable question as correct. Citation precision is the share of verified citations that point at a gold evidence page. 2–3 of 735 judge calls per run failed (invalid JSON after one retry); they are counted (`judge_failed`) and left out of the means and paired tests.

† **The faithfulness score is not reliable.** Against 50 blind human labels, `judge_v2` agrees on *correct* 46/50 (Cohen's κ 0.77) but on *faithful* only 40/50, κ 0.06: near chance level. The judge marked 7 of the 50 answers unfaithful, the human 5, and only 1 answer both. A high raw agreement here just reflects that both say "faithful" most of the time. So the column is shown for completeness, not as evidence; no conclusion in this README rests on it. The labels cover the `answer_v2` run, not the headline `answer_v3` run ([labels](results/human_labels/20261101T151804_answer_haiku_vlm_v2_7af24838e9f8.csv); `uv run docrag judge-agreement results/20261101T151804_answer_haiku_vlm_v2_7af24838e9f8 results/human_labels/20261101T151804_answer_haiku_vlm_v2_7af24838e9f8.csv`).

<details>
<summary><b>Retrieval table</b>: dense, BM25, hybrid, rerank, embedding size, OCR</summary>

Text-layer documents: 413 answerable questions; scanned: 152.

| variant | recall@1 | recall@10 | MRR@10 | recall@10, scanned PDFs |
|---|---|---|---|---|
| dense (gte-small) | 0.271 | 0.735 [0.672, 0.794] | 0.508 | 0.000 |
| BM25 (sparse + IDF) | 0.313 | 0.716 [0.656, 0.775] | 0.558 | 0.000 |
| hybrid (RRF, k=60) | 0.302 | 0.756 [0.702, 0.809] | 0.551 | 0.000 |
| hybrid + rerank | 0.330 | 0.752 [0.692, 0.808] | 0.563 | 0.000 |
| hybrid + rerank, no heading prefix | 0.327 | 0.747 [0.684, 0.803] | 0.558 | 0.000 |
| hybrid + rerank, gte-modernbert-base (768 dim) | 0.333 | 0.754 [0.695, 0.809] | 0.563 | 0.000 |
| hybrid + rerank + RapidOCR | 0.330 | 0.753 [0.694, 0.811] | 0.562 | 0.776 [0.658, 0.861] |
| hybrid + rerank + vision-LLM OCR | 0.334 | 0.755 [0.694, 0.812] | 0.564 | **0.834** [0.723, 0.911] |

Run folders: `results/*_dense_*`, `*_bm25_*`, `*_hybrid_*`, `*_baseline_*`, `*_no_prefix_*`, `*_modernbert_*`, `*_baseline_ocr_*`, `*_baseline_vlm_*` (the newest folder per name). These retrieval runs predate the commit recording in `run.json`. Their stored per-question metrics do match today's metric code applied to the stored rankings (checked by the script).
</details>

<details>
<summary><b>Experiments</b>: OCR, abstention, chatbot mode, answer-model comparison</summary>

- **OCR.** Without OCR the system can't see image-only PDFs (mostly slide decks). OCR takes recall@10 over all questions from 0.550 to 0.776, the biggest single gain in the project. The vision LLM beats RapidOCR mainly on charts (correct 0.220 → 0.309), because it binds numbers to their labels. Its OCR cost (0.42 EUR for 1,390 pages) was read from OpenRouter at the time and is not recorded in `results/`.
- **Abstention.** The first answer runs abstained on ~43% of answerable questions, although the gold page was often in the top 5. A new column (`abstained_before_llm`) showed the rerank gate stops only 2.8%; the LLM itself was saying "Not answerable". Four variants (`answer_v1`/`answer_v2` × 5/10 chunks) were compared on **dev**; only the winner ran on test. `answer_v2` + 10 chunks answers more (answerable +48/−7, p < 0.0001) but also answers questions whose premise is false (unanswerable +1/−18). `answer_v3` adds "answer exactly what was asked; false premise → *Not answerable*": unanswerable back to 0.929 (+25/−0 vs v2), answerable 0.430 (+3/−25 vs v2). v2 and v3 tie overall (+28/−25, p = 0.78). **How v3 came about:** its rules were written after reading v2's 18 lost unanswerable questions *on the test split*, then checked on dev (unanswerable 0.797 → 0.878, +6/−0, p = 0.031; overall n.s.), then run on test and picked as the headline with both test results in view. So v3's test numbers are not an independent held-out estimate, and the +3.6 points (v3 vs the vision-LLM baseline, +45/−19, p = 0.002) carry that selection effect. The cleanly held-out result is v2's: chosen on dev, run once on test, +3.2 points (+49/−25, p = 0.007).
- **Chatbot mode (search all documents).** All other numbers search only the question's own PDF. Asking across all 134 indexed PDFs (`scope: corpus`, [`configs/corpus_vlm.yaml`](configs/corpus_vlm.yaml), [run](results/20261101T174107_corpus_vlm_629974d14084/)) drops recall@10 from 0.776 to **0.599** [0.546, 0.655]: −0.177, CI [−0.211, −0.143] (text-layer 0.755 → 0.567, scanned 0.834 → 0.687). Answer quality in this mode is not benchmarked.
- **Answer-model comparison.** Same index, retrieval, prompt (`answer_v3`, 10 chunks) and judge; only the answer model changes (configs `answer_model_*.yaml`, runs `results/*_answer_model_*`). Paired McNemar against Haiku on the same 735 questions:

  | answer model | correct | vs Haiku | unanswerable | failed answers* | LLM p50 / p95 | EUR per 100 questions |
  |---|---|---|---|---|---|---|
  | **Claude Haiku 5.5** | **0.544** [0.501, 0.590] | – | 0.929 | 0.0% | 2.2 s / 7.4 s | **0.09** |
  | Gemini 3.8 Flash | 0.536 [0.493, 0.580] | −0.8 pts (p = 0.47, n.s.) | 0.935 | 0.1% | 4.6 s / 16 s | 0.51 |
  | Qwen3.8-27B | 0.521 [0.479, 0.566] | −2.3 pts (p = 0.036) | 0.839 | 6.8% | 5.0 s / 49 s | 0.37 |
  | Qwen3.8 Flash | 0.520 [0.475, 0.568] | −2.4 pts (p = 0.020) | 0.827 | 5.6% | 10.1 s / 68 s** | 0.14 |
  | GPT-5.4 mini | 0.473 [0.432, 0.516] | −7.2 pts (p < 0.001) | 0.667 | 0.0% | 1.4 s / 7.0 s | 0.26 |
  | Qwen3.5-9B | 0.434 [0.392, 0.478] | −11.0 pts (p < 0.001) | 0.875 | 0.0% | 0.9 s / 5.9 s | 0.01 |
  | Mistral Small 2603 | 0.423 [0.385, 0.462] | −12.2 pts (p < 0.001) | 0.780 | 0.0% | 1.0 s / 6.4 s | 0.07 |

  \* empty or invalid JSON after one retry, counted as wrong. The Qwen3.8 models are reasoning models: with the same 4,096-token limit as every other model, they sometimes spend it all on hidden reasoning and return nothing. Cost is `cost_eur_full` from each `run.json` (answers + judge, cache hits priced).
  \** from the first full run ([`results/_superseded/`](results/_superseded/README.md)); the re-run that replaced 4 rate-limited answers timed only 16 calls.

  Caveats: `answer_v3` was developed with Haiku, so other models may do better with their own prompt; there is one judge model. Haiku and Gemini 3.8 Flash are statistically tied; Gemini has the higher citation precision (0.810 vs 0.789) but costs about 5.6× as much. GPT-5.4 mini mostly loses on unanswerable questions (0.667 vs 0.929). Qwen3.5-9B often gets the answer right but rarely quotes verbatim: Verify rejected all citations of 93% of its non-abstained answers, so they come back as `unsupported`. That is exactly what Verify is for.
</details>

### How far to trust these numbers

- **Provenance of the headline run.** [`run.json`](results/20261101T162727_answer_haiku_vlm_v3_776cde4ae38c/run.json) of the `answer_v3` test run records `"commit": "826f8871328e68a9d04cc3a16173ec83a69b553f", "dirty": true`. Commit 826f887 exists in the history, but the run started with uncommitted changes under `src/`, `configs/`, `pyproject.toml` or `uv.lock`, and what they were was not recorded. Two traces show at least part of it: the run's config `configs/answer_haiku_vlm_v3.yaml` was first committed in 5d6b42b, and the `run.json` has `cost_eur_billed`/`cost_eur_full` fields that the 826f887 code cannot write (also added in 5d6b42b). Whether anything else differed is unknown. So these numbers can't be attributed to 826f887, nor to the current code, which has changed since. Other runs with `dirty: true`: `answer_haiku_vlm_v2` (b402121), `corpus_vlm` (ef2a510), and three model-comparison runs, `qwen35_9b`, `mistralsmall`, `qwen38_27b` (0102411). The retrieval runs record no commit at all. `scripts/reproduce_stats.py` prints this for every run.
- **What that limits, and what still holds.** You can't check out a commit and re-run to get the exact code behind those numbers. Re-running also needs paid API calls (the LLM cache is local, not committed), and hosted models can drift. What *is* reproducible is everything downstream of the per-question files: every mean, interval and test in this README.
- **No independent test validation for `answer_v3`.** It was designed from test-split errors and chosen with test results in view (see Experiments). There is no untouched test set left for that choice. A fresh held-out set (or a new split) would be needed to confirm it.
- **The faithfulness judge is at chance level** (κ 0.06, above). Correctness agreement (κ 0.77) is based on 50 labels from one human on one run.
- **One dataset, one judge model, one split.** The CIs cover resampling of documents, not the choice of judge, prompt or dataset.
- **From now on:** `docrag eval` refuses to start from a working tree with uncommitted result-relevant changes; `--allow-dirty` overrides it, logs a warning and records `"dirty": true, "allow_dirty": true` in `run.json`.

<details>
<summary>More notes: judge, costs, dataset quirks, latency</summary>

- **Judge reliability:** agreement with human labels is measured with `docrag label-sample` → `docrag label` → `docrag judge-agreement`; the 50 labels are in `results/human_labels/`. Faithfulness was hard for the human too: when retrieval missed part of the evidence, "every claim is in the cited passages" depends on how strictly the passages are read. A quote-only judge (`judge_v1`) put faithfulness at 0.46–0.50; with the full cited passages (`judge_v2`) it is 0.83–0.89, but given κ 0.06 neither level is validated.
- **Costs:** `run.json` reports `cost_eur_billed` (what was charged; cache hits are free) and `cost_eur_full` (cache hits priced from their stored usage, comparable across runs). The `answer_v3` test run: 0.67 EUR full. The project total of all LLM calls (~14 EUR) comes from the OpenRouter account, not from `results/`.
- **Dataset quirks:** 1,091 questions, 135 documents in the QA set, 134 PDFs downloadable (744 test questions in the split, 735 with a PDF). One question appears twice on the same document with contradictory gold answers ("46%" and "Not answerable"); paired tests align runs by row, not by question ID (the script checks that the rows line up).
- **Re-running from the cache** gives the same metrics; 2 of 735 answers were regenerated because retrieval returned a slightly different chunk set (probably GPU nondeterminism, not verified).
- **Latency (GPU, `answer_v3`):** search p50 6 ms, rerank p50 499 ms, LLM p50 2.2 s, whole answer p50 2.8 s / p95 6.2 s.
</details>

## Learnings

- **OCR only helps on PDFs without a text layer.** On text-layer documents recall@10 stays at 0.752 / 0.753 / 0.755 (no OCR / RapidOCR / vision LLM); on scanned ones it goes 0.000 → 0.776 → 0.834. Look at the data before tuning models. → [lesson 12](docs/lessons/12_ocr.md)
- **The LLM abstains, not the threshold.** Only 2.8% of answerable questions were stopped by the rerank gate. The fix was the prompt. → [lesson 08](docs/lessons/08_eval.md)
- **Check the judge before you trust the score.** `judge_v2` agrees with a human on correctness (κ 0.77) but its faithfulness verdicts are at chance level (κ 0.06). Earlier, averaging abstentions as "faithful" had made a metric artifact look like an OCR effect. → [lesson 08](docs/lessons/08_eval.md)
- **The more expensive model doesn't win.** Haiku and Gemini 3.8 Flash are statistically tied on correctness (0.544 vs 0.536, p = 0.47), but Gemini costs about 5.6× as much per question. → [lesson 05](docs/lessons/05_llm_proxy.md)
- **A small model can be right and still unsourced.** Verify rejected all citations of 93% of Qwen3.5-9B's answers because the quotes weren't verbatim. Without Verify those answers would look sourced. → [lesson 06](docs/lessons/06_answer_and_verify.md)

<details>
<summary>4 more learnings</summary>

- **Reranker, heading prefix and a 2× larger embedding model: no significant effect on retrieval.** Paired differences in recall@10 are within ±0.005, every CI includes 0; even the reranker's recall@1 gain (+0.028, CI [−0.013, +0.067]) is not significant. The reranker still costs ~0.5 s per query on GPU (p50 499 ms). → [lesson 04](docs/lessons/04_rerank_and_threshold.md)
- **Hybrid search beats BM25 (recall@10 +0.040, CI [+0.018, +0.066]) and dense at the top of the ranking** (MRR +0.044, CI [+0.015, +0.071]); against dense on recall@10 the gain is not significant. → [lesson 03](docs/lessons/03_embeddings_and_search.md)
- **Knowing the document is worth 18 points of recall.** Searching all 134 PDFs instead of the right one drops recall@10 from 0.776 to 0.599. → [lesson 03](docs/lessons/03_embeddings_and_search.md)
- **One number hides trade-offs.** v2 and v3 have the same overall correctness, but v2 gets 3.9 points more answerable questions right and v3 14.9 points more unanswerable ones. → [lesson 08](docs/lessons/08_eval.md)
</details>

## Quickstart

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), Docker for Qdrant. Works on Windows and Linux/macOS.

```bash
uv sync --extra cpu                          # or --extra gpu (CUDA wheels, no system CUDA needed)
uv run python scripts/reproduce_stats.py     # recompute every README number from results/ (free, offline)
uv run python scripts/download_data.py       # MMLongBench-Doc PDFs + questions -> data/
docker compose up -d qdrant && uv run docrag eval configs/baseline.yaml
```

`eval` indexes missing documents itself and prints the results table (`uv run docrag summary` prints it again, with CIs). It refuses to run from a working tree with uncommitted changes in `src/`, `configs/`, `pyproject.toml` or `uv.lock` (commit first, or pass `--allow-dirty`, which is recorded in `run.json`). Retrieval-only configs need no API key. For answers, set an LLM route in the config (`llm: provider:model`, e.g. `openrouter:anthropic/claude-haiku-5.5` with `OPENROUTER_API_KEY`); all LLM calls are cached, and a run stops at `DOCRAG_BUDGET_EUR` (default 5 EUR).

- **Ask one question:** `uv run docrag ask --config configs/answer_haiku_vlm_v3.yaml --doc COSTCO_2021_10K "What is total liabilities for costco in FY 2021?"`
- **API:** `uv run docrag serve --config configs/answer_haiku_vlm_v3.yaml` → http://127.0.0.1:8000/docs
- **Gradio demo locally:** `uv sync --extra cpu --extra demo`, then `uv run python demo/app.py` → http://127.0.0.1:7860
- **Web UI:** `cd ui && npm install && npm run build`, then `uv run docrag serve --config configs/answer_haiku_vlm_v3.yaml` → http://127.0.0.1:8000 (chat) and http://127.0.0.1:8000/#dashboard (every number computed live from `results/`). Or `docker compose up -d`, same URL. For UI development: `npm run dev` in `ui/` (port 5173, proxies the API).

<details>
<summary>More setup: LLM providers, Qdrant without Docker, tracing</summary>

| Option | Setup | Route example |
|---|---|---|
| OpenRouter (used for all results) | `OPENROUTER_API_KEY` | `openrouter:anthropic/claude-haiku-5.5` |
| Local (free) | install [Ollama](https://ollama.com), `ollama pull qwen3:8b` | `ollama:qwen3:8b` |
| OpenAI / Gemini / Groq / Mistral / DeepSeek | `OPENAI_API_KEY` / `GEMINI_API_KEY` / ... | `openai:<model>` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic:claude-haiku-5-5` |

Keys can go in a git-ignored `.env` or the OS keyring (`uv run keyring set docrag OPENROUTER_API_KEY`).

**Qdrant without Docker:** `DOCRAG_QDRANT_LOCATION=./qdrant_storage` uses the embedded mode. It's brute-force search, fine for a few documents.

**Everything in Docker:** `docker compose up -d` starts Qdrant, the API with the web UI on port 8000 and Phoenix on port 6006. Answers need `OPENROUTER_API_KEY` in `.env` (no OS keyring in the container).

**Tracing:** with `DOCRAG_OTEL_ENDPOINT=http://127.0.0.1:6006/v1/traces`, every `serve` request, `ask` call and `eval` question becomes an OpenTelemetry trace in Phoenix (embed, search, rerank, the LLM call with tokens and cost, Verify). Ingest is deliberately not traced.
</details>

<details>
<summary>How it's measured: experiment axes and metrics</summary>

One YAML in `configs/` = one row in the results table; the config hash is written into every result folder.

| Axis | Values | Config field |
|---|---|---|
| Embedding model | `gte-small` (384) vs `gte-modernbert-base` (768) | `embedder` |
| Retrieval | dense · BM25 · hybrid RRF · hybrid RRF + rerank | `retrieval` |
| OCR | none · RapidOCR · vision LLM | `ocr`, `ocr_model` |
| Heading prefix | heading chain in `embed_text` off · on | `prefix_headings` |
| Prompt, context size | `answer_v1`/`v2`/`v3`, 5 or 10 chunks | `prompt_version`, `context_k` |
| Verify | off · on (same answers, no extra LLM call) | reported side by side |

| Metric | Level | How |
|---|---|---|
| Recall@1/5/10, MRR@10, nDCG@10 | page | own helpers, tested against hand calculations |
| Citation precision | answer | share of citations on a gold page, Verify off vs on |
| Correct | answer | LLM judge (`judge_v2`, sees the cited passages); rules for "Not answerable" |
| Faithful | answer | same judge; not validated (κ 0.06 against human labels) |
| CIs, significance | run | bootstrap over documents (`eval/stats.py`); paired exact McNemar |
| Latency p50/p95, cost | run | `@timed` decorator (cache hits timed separately), usage from the proxy |
| Provenance | run | git commit + dirty flag in `run.json`; dirty runs refused unless `--allow-dirty` |
</details>

## Learn with this repo

Developed in an agentic development workflow. The lessons were built from that process: each explains one stage, and each notebook lets you rebuild a piece of it with the real modules (offline, fakes only, a few seconds per notebook).

<p align="center">
  <a href="docs/lessons/00_setup_and_tooling.md"><img alt="Lesson 00: Setup and tooling" src="docs/assets/readme/lesson-00.svg" width="13%"></a>
  <a href="docs/lessons/01_shared_foundations.md"><img alt="Lesson 01: Shared foundations" src="docs/assets/readme/lesson-01.svg" width="13%"></a>
  <a href="docs/lessons/02_ingest.md"><img alt="Lesson 02: Ingest" src="docs/assets/readme/lesson-02.svg" width="13%"></a>
  <a href="docs/lessons/03_embeddings_and_search.md"><img alt="Lesson 03: Embeddings and hybrid search" src="docs/assets/readme/lesson-03.svg" width="13%"></a>
  <a href="docs/lessons/04_rerank_and_threshold.md"><img alt="Lesson 04: Reranking and the threshold" src="docs/assets/readme/lesson-04.svg" width="13%"></a>
  <a href="docs/lessons/05_llm_proxy.md"><img alt="Lesson 05: The LLM proxy" src="docs/assets/readme/lesson-05.svg" width="13%"></a>
  <a href="docs/lessons/06_answer_and_verify.md"><img alt="Lesson 06: Answers and Verify" src="docs/assets/readme/lesson-06.svg" width="13%"></a>
  <br>
  <a href="docs/lessons/07_api_and_threads.md"><img alt="Lesson 07: API and threads" src="docs/assets/readme/lesson-07.svg" width="13%"></a>
  <a href="docs/lessons/08_eval.md"><img alt="Lesson 08: Evaluation" src="docs/assets/readme/lesson-08.svg" width="13%"></a>
  <a href="docs/lessons/09_testing_ci_docker.md"><img alt="Lesson 09: Testing, CI, Docker" src="docs/assets/readme/lesson-09.svg" width="13%"></a>
  <a href="docs/lessons/10_interview_prep.md"><img alt="Lesson 10: Interview prep" src="docs/assets/readme/lesson-10.svg" width="13%"></a>
  <a href="docs/lessons/11_gpu_secrets_security.md"><img alt="Lesson 11: GPU, secrets, security" src="docs/assets/readme/lesson-11.svg" width="13%"></a>
  <a href="docs/lessons/12_ocr.md"><img alt="Lesson 12: OCR" src="docs/assets/readme/lesson-12.svg" width="13%"></a>
  <a href="docs/lessons/13_tracing.md"><img alt="Lesson 13: Tracing" src="docs/assets/readme/lesson-13.svg" width="13%"></a>
</p>

<details>
<summary>All lessons: topics, Python concepts, notebooks, real modules</summary>

| # | Topic | Python concepts | Notebook | Real module |
|---|---|---|---|---|
| 00 | [Setup and tooling](docs/lessons/00_setup_and_tooling.md) | packages vs. modules, top-level `await` | [notebook](notebooks/lessons/00_setup_and_tooling.ipynb) | `wiring.py`, `cli.py` |
| 01 | [Shared foundations](docs/lessons/01_shared_foundations.md) | frozen Pydantic models, validators | [notebook](notebooks/lessons/01_shared_foundations.ipynb) | `shared/schema.py`, `shared/cache.py` |
| 02 | [Ingest](docs/lessons/02_ingest.md) | generators, `itertools.groupby` | [notebook](notebooks/lessons/02_ingest.ipynb) | `ingest/parser.py`, `ingest/chunker.py` |
| 03 | [Embeddings and hybrid search](docs/lessons/03_embeddings_and_search.md) | `Protocol`, registries | [notebook](notebooks/lessons/03_embeddings_and_search.ipynb) | `embed/`, `store/qdrant.py` |
| 04 | [Reranking and the threshold](docs/lessons/04_rerank_and_threshold.md) | `sorted(key=...)`, comprehensions | [notebook](notebooks/lessons/04_rerank_and_threshold.ipynb) | `rerank/reranker.py` |
| 05 | [The LLM proxy](docs/lessons/05_llm_proxy.md) | ABCs, decorators | [notebook](notebooks/lessons/05_llm_proxy.ipynb) | `llm/proxy.py` |
| 06 | [Answers and Verify](docs/lessons/06_answer_and_verify.md) | structured output, string handling | [notebook](notebooks/lessons/06_answer_and_verify.ipynb) | `answer/verify.py` |
| 07 | [API and threads](docs/lessons/07_api_and_threads.md) | FastAPI `TestClient`, locks | [notebook](notebooks/lessons/07_api_and_threads.ipynb) | `api/app.py`, `threads/store.py` |
| 08 | [Evaluation](docs/lessons/08_eval.md) | `statistics`, bootstrap | [notebook](notebooks/lessons/08_eval.ipynb) | `eval/metrics.py`, `eval/stats.py` |
| 09 | [Testing, CI, Docker](docs/lessons/09_testing_ci_docker.md) | pytest `parametrize` | [notebook](notebooks/lessons/09_testing_ci_docker.ipynb) | `tests/`, `ci.yml` |
| 10 | [Interview prep](docs/lessons/10_interview_prep.md) | – | – | everything |
| 11 | [GPU, secrets, security](docs/lessons/11_gpu_secrets_security.md) | `os.environ`, `SecretStr` | [notebook](notebooks/lessons/11_gpu_secrets_security.ipynb) | `shared/settings.py` |
| 12 | [OCR](docs/lessons/12_ocr.md) | `ThreadPoolExecutor`, the GIL | [notebook](notebooks/lessons/12_ocr.ipynb) | `ingest/ocr.py` |
| 13 | [Tracing](docs/lessons/13_tracing.md) | context managers | [notebook](notebooks/lessons/13_tracing.ipynb) | `shared/tracing.py` |

Solutions are in `notebooks/lessons/solutions/`; CI executes them so they stay in sync with the code.
</details>

## Known limitations

- **Measurement:** the headline run was made from uncommitted code; `answer_v3` was selected with test results in view; the faithfulness judge is at chance level; one dataset, one judge model, 50 human labels from one person (details in [How far to trust these numbers](#how-far-to-trust-these-numbers)).
- **Charts and figures are hard for a text retriever.** Only captions, labels and OCR/VLM descriptions are indexed; a ColPali-style visual retriever is future work.
- Verify checks that quotes are *real*, not that they *support* the claim (no entailment check).
- `answer_v2`/`v3` were developed with Haiku; other answer models may need their own prompt.
- Heading detection is a font-size heuristic; decorative text produces noisy heading chains.
- PyMuPDF is **AGPL-3.0**, which matters for closed-source redistribution.
- BM25 tokenization is English-only, without stemming; term IDs are `crc32` hashes (rare collisions possible).
- Threads are in-memory and lost on restart. Follow-up questions are not rewritten before retrieval. Searching all documents at once retrieves clearly worse than searching the right one (recall@10 0.599 vs 0.776).
- Qdrant's embedded mode is brute force and slow beyond roughly 20k chunks; use the Docker server for full runs.

<details>
<summary>Design decisions</summary>

- **PyMuPDF, not Docling.** It parses a 50-page PDF in seconds on CPU without ML models and detects ruled tables. Docling has better layout quality but needs PyTorch and is far slower. The parser sits behind a `Parser` protocol; a second parser is only added if a benchmark justifies it.
- **ONNX Runtime, not PyTorch.** About 50 MB instead of several GB, and the same model files run on CPU or CUDA. Pooling and truncation are implemented by hand, small and tested.
- **Hybrid search inside Qdrant.** Dense and BM25 prefetch branches fused with **RRF**, which only uses ranks, so cosine and BM25 scores never need calibrating. The RRF constant is explicit (`rrf_k: 60`), because Qdrant's default is k = 2.
- **BM25 = sparse vectors + `Modifier.IDF`.** Qdrant's full-text payload index only filters, it doesn't rank. TF and length normalization are computed at index time, IDF at query time.
- **Our own LLM proxy instead of LiteLLM.** Routing, format conversion, caching, cost accounting and the budget cap in about 300 tested lines. LiteLLM would be the pragmatic choice with many providers and could be swapped in behind the same `Provider` interface.
- **Structured output plus exactly one retry.** A Pydantic schema as `response_format`, one retry with the validation error, then a counted hard failure. No regex JSON repair.
- **Verify in pure code.** Deterministic, free, and runs on the same answers for the on/off comparison.
- **Tuned on dev, reported on test, split by document.** Questions about one PDF never end up in both splits. Thresholds, the `answer_v1`/`v2` × context-size grid and the judge were settled on dev. The exception is `answer_v3`, see [How far to trust these numbers](#how-far-to-trust-these-numbers).
- **Statistics over documents, tests paired.** Questions on one PDF are correlated, so CIs resample documents, not questions; two configs answer the same questions, so comparisons use exact McNemar on the disagreements.
- **Page-level metrics.** The ground truth is evidence pages, so ranked chunks are mapped to unique pages, which keeps chunkers comparable.
- **Versioned index.** The Qdrant collection name carries a fingerprint of parser, chunker, embedder and OCR versions; a change means a fresh index instead of silently reusing old chunks.
- **Tracing only on the query path.** `@timed` opens an OpenTelemetry span only *inside* a trace, so ingest records nothing. The `@timed` percentiles still feed `run.json`, so benchmark numbers don't depend on a tracing backend.
- **Content-addressed caches** (parse, OCR, embed, LLM): content hash + component version + options, never paths or secrets.
</details>

<details>
<summary>Future work</summary>

- **Next:** a held-out check of `answer_v3` on questions it was not selected on; a more reliable faithfulness check (a second judge model, or more human labels, on the headline run); a model-based support check (LettuceDetect) as a second Verify stage.
- **Later:** a judge panel across providers; Promptfoo as an eval gate in CI; Redis for threads and cache; a ColPali-style visual retriever; German as an evaluation axis.
</details>

<details>
<summary>Project layout</summary>

```
docrag-bench/
├── src/docrag/
│   ├── shared/     # settings, schema, cache, timing, tracing, IDs, config, provenance
│   ├── ingest/     # parser, OCR (RapidOCR, vision LLM), chunker, indexer
│   ├── embed/      # Embedder protocol, EmbeddingService, ONNX + fake
│   ├── store/      # Qdrant collections, BM25 sparse vectors, hybrid search, retriever
│   ├── rerank/     # cross-encoder + threshold calibration
│   ├── llm/        # proxy, pricing, providers (OpenAI-compatible, Anthropic, fake)
│   ├── answer/     # context, Answer schema, Verify, AnswerService
│   ├── threads/    # ThreadStore interface + in-memory implementation
│   ├── api/        # FastAPI routes (wiring only)
│   ├── eval/       # dataset, metrics, stats, judge, agreement, runner, calibration
│   ├── prompts/    # versioned prompt files
│   ├── wiring.py   # builds the object graph from settings + config
│   └── cli.py      # `docrag` command
├── scripts/        # data download, reproduce_stats.py (README numbers from results/)
├── demo/           # Gradio demo (local or Hugging Face Space)
├── ui/             # web UI: chat with sources + benchmark dashboard (Vite, React)
├── notebooks/lessons/  # interactive lessons (+ solutions/)
├── docs/lessons/       # lesson texts
├── tests/          # unit/ against fakes, e2e/ against a real Qdrant
├── configs/        # one YAML per experiment
├── results/        # per run: config.yaml, metrics.csv, per_question.csv, run.json
├── compose.yaml    # Qdrant + API + Phoenix
└── pyproject.toml  # uv, ruff, mypy (strict), pytest
```
</details>

<details>
<summary>Development</summary>

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src                                   # strict mode
uv run pytest -m "not e2e"                        # unit tests, no services needed
uv run pytest --nbmake notebooks/lessons/solutions # lesson notebooks
DOCRAG_E2E_QDRANT_URL=http://localhost:6333 uv run pytest -m e2e   # needs Qdrant
uv run python scripts/reproduce_stats.py          # README statistics from results/
```

CI (GitHub Actions) runs lint, types, unit tests and the lesson notebooks; an end-to-end test against a Qdrant service container; and a Docker build.
</details>

## License

The code is MIT licensed. **Data:** the MMLongBench-Doc dataset card on Hugging Face does not state a license; the PDFs come from various public sources and are downloaded by `scripts/download_data.py`, not redistributed here. The models (`gte-small`: MIT; `gte-modernbert-base`, `gte-reranker-modernbert-base`: Apache-2.0) are downloaded at runtime.
