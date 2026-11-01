# docrag-bench web UI

Two views on top of the FastAPI backend, Vite + React + TypeScript, no router or state library:

- **Chat with sources** (`#chat`): pick an indexed PDF, or **All documents** to ask across every PDF like a company chatbot (retrieval is clearly weaker in that mode: recall@10 0.599 vs 0.776, see the main README), and ask questions in threads. Answers show clickable `[p.X]` citation chips and the verified quotes; the right panel lists the retrieved evidence after reranking (rank, page, TABLE/TEXT/FIGURE/OCR, rerank score, cited) with a preview of the selected page. "Not answerable" is shown with its reason (rerank gate or the model). The footer shows latency, tokens and the uncached cost of the last answer.
- **Benchmark dashboard** (`#dashboard`): runs table with 95% CIs, recall@10 / correct chart, text-layer vs. scanned documents, the abstention experiment and judge-vs-human agreement. Everything comes from `GET /results`, which reads `results/`; nothing is hard-coded.

Threads are kept in the browser (localStorage); the backend keeps the conversation history it needs for the LLM.

## Development

```bash
uv run docrag serve --config configs/answer_haiku_vlm_v3.yaml   # backend on :8000
cd ui && npm install && npm run dev                            # UI on :5173, API calls proxied to :8000
```

## Production

```bash
cd ui && npm run build    # -> ui/dist
uv run docrag serve --config configs/answer_haiku_vlm_v3.yaml   # serves ui/dist at http://127.0.0.1:8000
```

Or `docker compose up -d`: the image builds the UI in a Node stage and serves it from the same container as the API.

Answers need an LLM key (`OPENROUTER_API_KEY`); the dashboard and the evidence views of earlier answers don't.
