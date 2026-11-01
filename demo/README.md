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

It runs the same pipeline the [benchmark](https://github.com/byQuexo/docrag-bench) measures: hybrid search, cross-encoder rerank, abstain gate, LLM, Verify, with the headline prompt `answer_v3`. Scanned pages are read with local RapidOCR (free, CPU) instead of the vision LLM, and Qdrant runs in memory.

## Who answers

Chosen at startup; the page shows which mode is active.

| Mode | Set | Cost for the Space owner | Quality |
|---|---|---|---|
| **Open model on ZeroGPU** (recommended for a public Space) | variable `DOCRAG_DEMO_LOCAL_MODEL=Qwen/Qwen3.8-27B-FP8` | none: GPU time comes from each visitor's free daily ZeroGPU quota | benchmarked via OpenRouter: correct 0.521 vs 0.544 for Haiku, see below |
| Claude Haiku 5.5 | secret `OPENROUTER_API_KEY` | ~0.1 EUR per 100 questions, capped by `DOCRAG_BUDGET_EUR` | the benchmark headline (0.544) |
| Retrieval only | nothing | none | no answers: shows the ranked evidence and the best page |

**Why Qwen3.8-27B:** in the [answer-model comparison](https://github.com/byQuexo/docrag-bench#results) it came closest to Haiku among the open models (correct 0.521 [0.479, 0.566]). The smaller Qwen3.5-9B scored 0.434, and Verify rejected the citations of 93% of its answers (it rarely copies quotes verbatim), so it would mark almost every answer as unsourced. The FP8 version (~28 GB) fits ZeroGPU's 48 GB slot.

**Honest limits of the ZeroGPU mode**
- The benchmark ran Qwen3.8 on OpenRouter, with its built-in reasoning and server-side JSON enforcement. In the Space, reasoning is turned off (`enable_thinking=False`, otherwise it can use up the token budget and return nothing) and the JSON format comes from the prompt plus one retry. Both can change the quality; this exact setup is not benchmarked.
- The FP8 weights need a GPU with FP8 support (ZeroGPU's Blackwell GPUs have it). It could not be tested locally; the code path itself was tested with the small `Qwen/Qwen3.5-0.8B` on CPU.
- After each restart the Space loads ~28 GB of weights, so the first start takes a while.

## Deploy as a Hugging Face Space (ZeroGPU)

Hosting a ZeroGPU Space is free for personal accounts with a verified email that are older than 30 days (up to 2 Spaces).

1. Create a Space: SDK **Gradio**, hardware **ZeroGPU**.
2. Upload the contents of this `demo/` folder: `README.md` (this file, its header configures the Space), `app.py`, `local_llm.py`, `config.yaml`, `requirements.txt`.
3. Add sample PDFs in a `samples/` folder, e.g. `fixtures/tiny.pdf` and `fixtures/tiny_scanned.pdf` from the repo. Only upload PDFs you may redistribute (the benchmark PDFs have no clear license).
4. Settings → Variables: `DOCRAG_DEMO_LOCAL_MODEL` = `Qwen/Qwen3.8-27B-FP8`. No secret needed.

If the 27B model fails to load, set `DOCRAG_DEMO_LOCAL_MODEL=Qwen/Qwen3.5-9B` (bf16, safe to load, but weaker as described above), or remove the variable for retrieval-only mode.

Optional speed-up for Qwen3.5/3.8 on GPU: the `flash-linear-attention` and `causal-conv1d` kernels (without them `transformers` uses a slower but correct fallback). They are left out of `requirements.txt` because they compile CUDA code at build time.

## Run locally

```bash
uv sync --extra cpu --extra demo        # or --extra gpu
uv run python demo/app.py               # http://127.0.0.1:7860
```

Locally the demo uses Haiku if `OPENROUTER_API_KEY` is available (environment, `.env` or the OS keyring), otherwise retrieval only. To try the open-model path locally, install `torch transformers spaces accelerate` and set `DOCRAG_DEMO_LOCAL_MODEL` (a small model like `Qwen/Qwen3.5-0.8B` runs on CPU).

## Limits

| Limit | Default | Environment variable |
|---|---|---|
| LLM budget per process (Haiku mode, all visitors together) | 1 EUR | `DOCRAG_BUDGET_EUR` |
| Questions per browser session | 10 | `DOCRAG_DEMO_MAX_QUESTIONS` |
| Upload size | 15 MB | `DOCRAG_DEMO_MAX_UPLOAD_MB` |
| Pages per uploaded PDF | 60 | `DOCRAG_DEMO_MAX_PAGES` |
| New tokens per answer (open model) | 1024 | `DOCRAG_DEMO_MAX_NEW_TOKENS` |

Answers are cached, so a repeated question costs nothing. Requests are handled one at a time (the in-memory index is shared).
