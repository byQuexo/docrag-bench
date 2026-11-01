---
title: docrag-bench
emoji: 📄
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: PDF question answering with verified citations
---

# docrag-bench demo

Pick a sample PDF or upload one, ask a question, and get an answer whose citations are checked: the cited passage must have been retrieved, and the quote must appear in it verbatim. If the document doesn't contain the answer, the demo says **"Not answerable"** instead of guessing.

It runs the same pipeline the [benchmark](https://github.com/byQuexo/docrag-bench) measures (hybrid search, cross-encoder rerank, abstain gate, LLM, Verify) with the headline prompt `answer_v3`. Two differences from the benchmarked setup: scanned pages are read with local RapidOCR instead of a vision LLM (free, CPU), and Qdrant runs in memory. This exact combination was not benchmarked as a whole.

## Run locally

```bash
uv sync --extra cpu --extra demo        # or --extra gpu
uv run python demo/app.py               # http://127.0.0.1:7860
```

Needs an OpenRouter key (`OPENROUTER_API_KEY` in the environment, `.env` or the OS keyring). Samples: the repo's `fixtures/*.pdf`, plus the benchmark PDFs if you downloaded them.

## Deploy as a Hugging Face Space

1. Create a Space (SDK: Gradio, hardware: CPU basic is enough).
2. Upload the contents of this `demo/` folder: `README.md` (this file, its header configures the Space), `app.py`, `config.yaml`, `requirements.txt`.
3. Add sample PDFs in a `samples/` folder, e.g. `fixtures/tiny.pdf` and `fixtures/tiny_scanned.pdf` from the repo. Only upload PDFs you may redistribute (the benchmark PDFs have no clear license).
4. Settings → Secrets: `OPENROUTER_API_KEY`.

`requirements.txt` installs `docrag-bench` from GitHub, so the repository has to be public first.

## Cost protection

| Limit | Default | Environment variable |
|---|---|---|
| LLM budget per process (all visitors together) | 1 EUR | `DOCRAG_BUDGET_EUR` |
| Questions per browser session | 10 | `DOCRAG_DEMO_MAX_QUESTIONS` |
| Upload size | 15 MB | `DOCRAG_DEMO_MAX_UPLOAD_MB` |
| Pages per uploaded PDF | 60 | `DOCRAG_DEMO_MAX_PAGES` |

Answers are cached, so a repeated question costs nothing. Requests are handled one at a time (the in-memory index is shared). On CPU, the first question about a new PDF also downloads the ONNX models and indexes the file, so expect it to take a while; reranking takes seconds per question on CPU.
