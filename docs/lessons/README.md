# docrag-bench lessons

These lessons walk through the code of docrag-bench one stage at a time. The code
is real and runs; every snippet below is copied from `src/docrag/`. The goal is
that afterwards you can (1) write this kind of Python yourself and (2) defend every
design decision in an interview.

## Lessons

| # | Lesson | Notebook | Code it covers | Plan milestone |
|---|--------|----------|----------------|----------------|
| 00 | [Setup and tooling](00_setup_and_tooling.md) | [exercise](../../notebooks/lessons/00_setup_and_tooling.ipynb) · [solution](../../notebooks/lessons/solutions/00_setup_and_tooling.ipynb) | `pyproject.toml`, `uv.lock`, src layout, `cli.py` | M0 |
| 01 | [Shared foundations](01_shared_foundations.md) | [exercise](../../notebooks/lessons/01_shared_foundations.ipynb) · [solution](../../notebooks/lessons/solutions/01_shared_foundations.ipynb) | `shared/*` | M0 |
| 02 | [Ingest: PDF -> Blocks -> Chunks](02_ingest.md) | [exercise](../../notebooks/lessons/02_ingest.ipynb) · [solution](../../notebooks/lessons/solutions/02_ingest.ipynb) | `ingest/*`, `shared/ids.py` | M1 |
| 03 | [Embeddings and hybrid search](03_embeddings_and_search.md) | [exercise](../../notebooks/lessons/03_embeddings_and_search.ipynb) · [solution](../../notebooks/lessons/solutions/03_embeddings_and_search.ipynb) | `embed/*`, `store/*` | M2 |
| 04 | [Reranking and the threshold](04_rerank_and_threshold.md) | [exercise](../../notebooks/lessons/04_rerank_and_threshold.ipynb) · [solution](../../notebooks/lessons/solutions/04_rerank_and_threshold.ipynb) | `rerank/reranker.py`, `eval/calibrate.py` | M3 |
| 05 | [The LLM proxy](05_llm_proxy.md) | [exercise](../../notebooks/lessons/05_llm_proxy.ipynb) · [solution](../../notebooks/lessons/solutions/05_llm_proxy.ipynb) | `llm/*` | M3 |
| 06 | [Answers and Verify](06_answer_and_verify.md) | [exercise](../../notebooks/lessons/06_answer_and_verify.ipynb) · [solution](../../notebooks/lessons/solutions/06_answer_and_verify.ipynb) | `answer/*`, `prompts/*` | M3 |
| 07 | [API and threads](07_api_and_threads.md) | [exercise](../../notebooks/lessons/07_api_and_threads.ipynb) · [solution](../../notebooks/lessons/solutions/07_api_and_threads.ipynb) | `api/app.py`, `threads/store.py`, `wiring.py` | M4 |
| 08 | [Evaluation](08_eval.md) | [exercise](../../notebooks/lessons/08_eval.ipynb) · [solution](../../notebooks/lessons/solutions/08_eval.ipynb) | `eval/*`, `configs/*` | M5 |
| 09 | [Testing, CI, Docker](09_testing_ci_docker.md) | [exercise](../../notebooks/lessons/09_testing_ci_docker.ipynb) · [solution](../../notebooks/lessons/solutions/09_testing_ci_docker.ipynb) | `tests/*`, `.github/workflows/ci.yml`, `Dockerfile`, `compose.yaml` | M0/M4 |
| 10 | [Interview prep](10_interview_prep.md) | - | everything | - |
| 11 | [GPU, secrets, security, OpenRouter](11_gpu_secrets_security.md) | [exercise](../../notebooks/lessons/11_gpu_secrets_security.ipynb) · [solution](../../notebooks/lessons/solutions/11_gpu_secrets_security.ipynb) | `embed/onnx.py`, `shared/settings.py`, `llm/openai_compat.py`, `compose.yaml`, `pyproject.toml` | after M5 |
| 12 | [OCR for PDFs without a text layer](12_ocr.md) | [exercise](../../notebooks/lessons/12_ocr.ipynb) · [solution](../../notebooks/lessons/solutions/12_ocr.ipynb) | `ingest/ocr.py`, `ingest/indexer.py`, `ingest/chunker.py`, `eval/runner.py` | after M5 |
| 13 | [Tracing with OpenTelemetry + Phoenix](13_tracing.md) | [exercise](../../notebooks/lessons/13_tracing.ipynb) · [solution](../../notebooks/lessons/solutions/13_tracing.ipynb) | `shared/tracing.py`, `shared/timing.py`, `api/app.py`, `eval/runner.py`, `compose.yaml` | M6 |

## Interactive notebooks

Every lesson except 10 has a notebook in `notebooks/lessons/`: 2-4 small steps where
you type 1-5 lines yourself (`# TODO`), each followed by a check cell with `assert`s
that give a hint when something's off. They import the real `docrag` code, use only
`fixtures/` and the fake providers, and run offline in seconds. Solutions are in
`notebooks/lessons/solutions/` (CI executes those, so they stay in sync with the code).

Open them in PyCharm or VS Code and pick the project's `.venv` as the kernel
(`ipykernel` is in the dev dependencies). Run the solutions headless with
`uv run pytest --nbmake notebooks/lessons/solutions`.

## How to use them

For each lesson:

1. **Read the lesson** once, top to bottom.
2. **Open the files it names** in PyCharm next to the lesson. Follow the
   "step by step" section with the code open. Set a breakpoint and run the
   matching test (`uv run pytest tests/unit/test_<module>.py`) to see real values.
3. **Do the exercises.** Each one says when it is done. Revert with
   `git checkout -- <file>` when you're finished.
4. **Answer the interview questions out loud** before reading the model answers.
   If you can't explain something in 3 sentences, reread that part of the code.

Lesson 10 collects 60+ interview questions across the whole project. Go through it
again the evening before an interview.

## Quick command reference

```powershell
uv sync --extra gpu   # or --extra cpu; installs exactly what uv.lock says
uv run pytest                                  # all tests (e2e skipped without Qdrant)
uv run ruff check . ; uv run mypy src          # lint + types
docker compose up -d qdrant                    # start Qdrant on :6333
uv run docrag ingest --config configs/baseline.yaml --docs 10
uv run docrag calibrate --config configs/baseline.yaml
uv run docrag eval configs/retrieval_*.yaml configs/baseline.yaml
uv run docrag summary
uv run docrag ask --config configs/baseline.yaml --doc PH_2016.06.08_Economy-Final "How many Latinos were surveyed?"
uv run docrag serve --config configs/baseline.yaml   # then open http://127.0.0.1:8000/docs
docker compose up -d phoenix                   # trace viewer on :6006
$env:DOCRAG_OTEL_ENDPOINT = "http://127.0.0.1:6006/v1/traces"   # tracing on for serve/ask/eval
```
