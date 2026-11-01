// Types mirror the FastAPI responses (src/docrag/api/app.py, eval/overview.py).

export type Evidence = {
  rank: number;
  id: string;
  doc_id: string;
  page: number;
  type: "TEXT" | "TABLE" | "FIGURE" | "OCR";
  score: number;
  cited: boolean;
  text: string;
};

export type Source = { id: string; doc_id: string; page: number; quote: string; page_image_url: string };

export type QueryResponse = {
  thread_id: string;
  answer: string;
  status: "supported" | "partially_supported" | "unsupported" | "abstained";
  sources: Source[];
  rejected_citations: number;
  evidence: Evidence[];
  abstained_before_llm: boolean;
  top_score: number | null;
  threshold: number | null;
  input_tokens: number;
  output_tokens: number;
  cost_eur: number;
  latency_ms: number;
};

export type Interval = { mean: number; lo: number; hi: number };
export type Part = { questions: number; recall_at_10: number | null; correct: number | null };

export type Run = {
  name: string;
  folder: string;
  config_hash: string | null;
  split: string | null;
  commit: string | null;
  setup: {
    retrieval: string | null;
    scope: string | null;
    embedder: string | null;
    ocr: string | null;
    ocr_model: string | null;
    prefix_headings: boolean | null;
    prompt_version: string | null;
    context_k: number | null;
    llm: string | null;
    judge_prompt_version: string | null;
  };
  n: number;
  recall_at_10: Interval | null;
  mrr_at_10: number | null;
  correct: Interval | null;
  correct_unanswerable: number | null;
  abstained: number | null;
  abstained_answerable: number | null;
  gate_answerable: number | null;
  faithful: number | null;
  citation_precision: number | null;
  text_layer: Part;
  scanned: Part;
  cost_eur_full: number | null;
  latency_p50_ms: Record<string, number>;
};

export type Agreement = { n: number; agree: number; kappa: number | null };

export type Overview = {
  split: { questions: number; documents: number; scanned_documents: number } | null;
  runs: Run[];
  judge_agreement: { run: string; labels: Record<string, Agreement> }[];
};

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  documents: () => fetch("/documents").then((r) => json<string[]>(r)),
  results: () => fetch("/results").then((r) => json<Overview>(r)),
  /** docId null = search every indexed document (chatbot mode). */
  query: (question: string, docId: string | null, threadId: string | null) =>
    fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, doc_ids: docId ? [docId] : null, thread_id: threadId }),
    }).then((r) => json<QueryResponse>(r)),
  pageUrl: (docId: string, page: number) => `/pages/${encodeURIComponent(docId)}/${page}.png`,
};

export const fmt = (x: number | null | undefined, digits = 3) =>
  x === null || x === undefined ? "–" : x.toFixed(digits);
