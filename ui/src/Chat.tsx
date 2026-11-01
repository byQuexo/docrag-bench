import { Fragment, useEffect, useMemo, useState } from "react";
import { api, fmt, type Evidence, type QueryResponse } from "./api";

type Turn = { question: string; response?: QueryResponse; error?: string };
type Thread = { id: string; doc: string; title: string; turns: Turn[]; backendId: string | null };

// Threads live in the browser (localStorage); the backend keeps the history it
// needs for the LLM under backendId.
const KEY = "docrag.threads";
const load = (): Thread[] => {
  try {
    return JSON.parse(localStorage.getItem(KEY) ?? "[]") as Thread[];
  } catch {
    return [];
  }
};

const CITE = /\[([^\]\s]+):p(\d+):c\d+\]/g;

// Picking this searches every indexed document (chatbot mode, scope "corpus").
const ALL = "*";
const docLabel = (doc: string) => (doc === ALL ? "All documents" : doc);

/** Answer text with [doc:p12:c1] markers turned into clickable [p.12] chips. */
function AnswerText({ text, onCite }: { text: string; onCite: (id: string) => void }) {
  const parts: (string | { id: string; page: string })[] = [];
  let last = 0;
  for (const m of text.matchAll(CITE)) {
    parts.push(text.slice(last, m.index));
    parts.push({ id: m[0].slice(1, -1), page: m[2] });
    last = (m.index ?? 0) + m[0].length;
  }
  parts.push(text.slice(last));
  return (
    <p className="answer-text">
      {parts.map((p, i) =>
        typeof p === "string" ? (
          <Fragment key={i}>{p}</Fragment>
        ) : (
          <button key={i} className="chip" onClick={() => onCite(p.id)}>
            p.{p.page}
          </button>
        ),
      )}
    </p>
  );
}

function AnswerView({ r, onCite, showDoc }: { r: QueryResponse; onCite: (id: string) => void; showDoc: boolean }) {
  if (r.status === "abstained") {
    const why = r.abstained_before_llm
      ? `top rerank score ${fmt(r.top_score, 2)} < threshold ${fmt(r.threshold, 2)}: no LLM call`
      : "the model found no answer to exactly this question in the retrieved passages";
    return (
      <div className="abstain">
        <strong>Not answerable from this document</strong>
        <p>It abstains instead of guessing.</p>
        <code>abstain · {why}</code>
      </div>
    );
  }
  const verify = {
    supported: "every citation verified",
    partially_supported: `${r.rejected_citations} unverifiable citation(s) removed`,
    unsupported: "no citation could be verified: unsourced",
  }[r.status];
  return (
    <div className="answer">
      <AnswerText text={r.answer} onCite={onCite} />
      <div className="sources">
        {r.sources.map((s) => (
          <button key={s.id + s.quote} className="source" onClick={() => onCite(s.id)}>
            <span className="source-page">
              {showDoc && <span className="muted">{s.doc_id} · </span>}p.{s.page}
            </span>
            <span className="source-quote">“{s.quote}”</span>
          </button>
        ))}
      </div>
      <div className={`verify verify-${r.status}`}>Verify: {verify}</div>
    </div>
  );
}

export function Chat() {
  const [docs, setDocs] = useState<string[]>([]);
  const [threads, setThreads] = useState<Thread[]>(load);
  const [activeId, setActiveId] = useState<string | null>(() => load()[0]?.id ?? null);
  const [doc, setDoc] = useState("");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [docsError, setDocsError] = useState<string | null>(null);

  useEffect(() => {
    api.documents().then(setDocs, (e: Error) => setDocsError(e.message));
  }, []);
  useEffect(() => localStorage.setItem(KEY, JSON.stringify(threads)), [threads]);

  const active = threads.find((t) => t.id === activeId) ?? null;
  const lastResponse = active?.turns.findLast((t) => t.response)?.response;
  const evidence: Evidence[] = lastResponse?.evidence ?? [];
  const shown = useMemo(
    () => evidence.find((e) => e.id === selected) ?? evidence.find((e) => e.cited) ?? evidence[0],
    [evidence, selected],
  );

  const update = (id: string, fn: (t: Thread) => Thread) =>
    setThreads((ts) => ts.map((t) => (t.id === id ? fn(t) : t)));

  function newThread() {
    setActiveId(null);
    setSelected(null);
  }

  async function ask() {
    const q = question.trim();
    const target = active?.doc ?? doc;
    if (!q || !target || busy) return;
    let thread = active;
    if (!thread) {
      thread = { id: crypto.randomUUID(), doc: target, title: q.slice(0, 60), turns: [], backendId: null };
      const created = thread;
      setThreads((ts) => [created, ...ts]);
      setActiveId(created.id);
    }
    const id = thread.id;
    update(id, (t) => ({ ...t, turns: [...t.turns, { question: q }] }));
    setQuestion("");
    setBusy(true);
    setSelected(null);
    try {
      const r = await api.query(q, target === ALL ? null : target, thread.backendId);
      update(id, (t) => ({
        ...t,
        backendId: r.thread_id,
        turns: t.turns.map((turn, i) => (i === t.turns.length - 1 ? { ...turn, response: r } : turn)),
      }));
    } catch (e) {
      const message = (e as Error).message;
      update(id, (t) => ({
        ...t,
        turns: t.turns.map((turn, i) => (i === t.turns.length - 1 ? { ...turn, error: message } : turn)),
      }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat">
      <aside className="sidebar">
        <button className="new-thread" onClick={newThread}>
          + New thread
        </button>
        <div className="label">Threads</div>
        {threads.length === 0 && <div className="muted small">No threads yet.</div>}
        {threads.map((t) => (
          <button
            key={t.id}
            className={`thread ${t.id === activeId ? "active" : ""}`}
            onClick={() => {
              setActiveId(t.id);
              setSelected(null);
            }}
          >
            <span>{t.title}</span>
            <span className="muted small">
              {docLabel(t.doc)} · {t.turns.length} turn{t.turns.length === 1 ? "" : "s"}
            </span>
          </button>
        ))}
      </aside>

      <main className="conversation">
        <div className="conv-header">
          {active ? (
            <>
              <strong>{active.title}</strong>
              <span className="pill">{docLabel(active.doc)}</span>
            </>
          ) : (
            <label className="doc-pick">
              Document
              <select value={doc} onChange={(e) => setDoc(e.target.value)}>
                <option value="">Choose an indexed PDF…</option>
                <option value={ALL}>All documents (search every PDF, like a chatbot)</option>
                {docs.map((d) => (
                  <option key={d}>{d}</option>
                ))}
              </select>
              {docsError && <span className="error small">{docsError}</span>}
            </label>
          )}
        </div>
        <div className="messages">
          {active?.turns.map((turn, i) => (
            <Fragment key={i}>
              <div className="bubble">{turn.question}</div>
              {turn.response && (
                <AnswerView r={turn.response} onCite={setSelected} showDoc={active.doc === ALL} />
              )}
              {turn.error && <div className="error">Request failed: {turn.error}</div>}
              {!turn.response && !turn.error && <div className="muted">Retrieving, reranking, answering…</div>}
            </Fragment>
          ))}
          {!active && (
            <div className="muted empty">
              Pick a document, or "All documents" to ask across every PDF like a company chatbot.
            </div>
          )}
        </div>
        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            void ask();
          }}
        >
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask a question about this document…"
          />
          <button disabled={busy || !question.trim() || !(active?.doc ?? doc)}>Send</button>
        </form>
        <footer className="usage">
          {lastResponse ? (
            <>
              latency <b>{(lastResponse.latency_ms / 1000).toFixed(1)} s</b> · cost / answer{" "}
              <b>€{lastResponse.cost_eur.toFixed(4)}</b> (uncached price) · tokens{" "}
              <b>
                {lastResponse.input_tokens} in · {lastResponse.output_tokens} out
              </b>
            </>
          ) : (
            <span className="muted">Latency and cost of the last answer appear here.</span>
          )}
        </footer>
      </main>

      <aside className="evidence">
        <div className="label">Retrieved evidence · top-k after rerank</div>
        {evidence.length === 0 && <div className="muted small">Evidence for the last answer appears here.</div>}
        {evidence.map((e) => (
          <button
            key={e.id}
            className={`ev ${e.id === shown?.id ? "active" : ""}`}
            onClick={() => setSelected(e.id)}
          >
            <div className="ev-head">
              <span>{e.rank}</span>
              <b>
                {active?.doc === ALL && <span className="muted">{e.doc_id.slice(0, 18)} · </span>}p.{e.page}
              </b>
              <span className={`type type-${e.type}`}>{e.type}</span>
              {e.cited && <span className="cited">cited</span>}
              <span className="score">rerank {e.score.toFixed(2)}</span>
            </div>
            <div className="ev-text">{e.text}</div>
            <div className="bar">
              <div style={{ width: `${Math.max(0, Math.min(1, e.score)) * 100}%` }} />
            </div>
          </button>
        ))}
        {shown && (
          <div className="preview">
            <img src={api.pageUrl(shown.doc_id, shown.page)} alt={`page ${shown.page}`} />
            <div className="small muted">
              {shown.doc_id} · page {shown.page} · {shown.type} · rerank {shown.score.toFixed(2)}
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
