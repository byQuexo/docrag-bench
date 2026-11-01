import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, ErrorBar, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, fmt, type Interval, type Overview, type Run } from "./api";

// Every number on this page comes from GET /results, which reads results/.

const ci = (x: Interval | null) =>
  x ? (
    <>
      {x.mean.toFixed(3)} <span className="ci">[{x.lo.toFixed(3)}, {x.hi.toFixed(3)}]</span>
    </>
  ) : (
    "–"
  );

function setupLabel(r: Run) {
  const s = r.setup;
  const parts = [s.retrieval ?? "", s.ocr && s.ocr !== "none" ? `OCR: ${s.ocr}` : "no OCR"];
  if (s.llm) parts.push(`${s.prompt_version} · ${s.context_k} chunks · ${s.llm.split(/[:/]/).pop()}`);
  if (s.embedder && !s.embedder.endsWith("gte-small")) parts.push(s.embedder);
  if (s.prefix_headings === false) parts.push("no heading prefix");
  if (s.scope === "corpus") parts.push("searches ALL documents");
  return parts.join(" · ");
}

function Stat({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="small muted">{sub}</div>}
    </div>
  );
}

export function Dashboard() {
  const [data, setData] = useState<Overview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDev, setShowDev] = useState(false);

  useEffect(() => {
    api.results().then(setData, (e: Error) => setError(e.message));
  }, []);

  if (error) return <div className="page error">Could not load results: {error}</div>;
  if (!data) return <div className="page muted">Loading results…</div>;

  const runs = data.runs.filter((r) => showDev || r.split !== "dev");
  const test = data.runs.filter((r) => r.split === "test");
  const answerRuns = test.filter((r) => r.correct);
  const best = answerRuns.reduce<Run | null>((b, r) => (!b || r.correct!.mean > b.correct!.mean ? r : b), null);
  // Retrieval runs only: answer runs on the same index differ just by GPU jitter.
  const retrievalRuns = test.filter((r) => !r.setup.llm && r.setup.scope !== "corpus");
  const bestRecall = retrievalRuns.reduce<Run | null>(
    (b, r) => (!b || (r.recall_at_10?.mean ?? 0) > (b.recall_at_10?.mean ?? 0) ? r : b),
    null,
  );
  const agreement = data.judge_agreement[0];
  // The pipeline progression with the benchmark's answer model (Haiku); the
  // answer-model comparison is in the table.
  const haikuRuns = answerRuns.filter((r) => (r.setup.llm ?? "").includes("claude-haiku"));
  const chart = haikuRuns.map((r) => ({
    name: r.name.replace("answer_haiku", "haiku").replace("haiku_", ""),
    "recall@10": r.recall_at_10?.mean,
    recallErr: r.recall_at_10 ? [r.recall_at_10.mean - r.recall_at_10.lo, r.recall_at_10.hi - r.recall_at_10.mean] : 0,
    correct: r.correct?.mean,
    correctErr: r.correct ? [r.correct.mean - r.correct.lo, r.correct.hi - r.correct.mean] : 0,
  }));
  // Text layer vs. scanned: retrieval runs that differ only in OCR.
  const ocrRuns = test.filter(
    (r) =>
      !r.setup.llm &&
      r.setup.retrieval === "hybrid_rerank" &&
      r.setup.scope !== "corpus" &&
      r.setup.prefix_headings !== false &&
      (r.setup.embedder ?? "").endsWith("gte-small"),
  );
  // Abstention experiment: answer runs on the vision-LLM index.
  const abstention = haikuRuns.filter((r) => r.setup.ocr === "vlm");

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1>Benchmark runs</h1>
          <div className="muted">
            MMLongBench-Doc · test split · {data.split?.questions} questions · {data.split?.documents} documents ·{" "}
            {data.split?.scanned_documents} scanned
          </div>
        </div>
        <span className="pill">[ ] = 95% CI, bootstrap over documents</span>
      </div>

      <div className="stats">
        <Stat label="Questions" value={data.split?.questions} sub={`${data.split?.documents} docs · ${data.split?.scanned_documents} scanned`} />
        <Stat label="Best recall@10" value={ci(bestRecall?.recall_at_10 ?? null)} sub={bestRecall?.name} />
        <Stat label="Best correct" value={ci(best?.correct ?? null)} sub={best?.name} />
        <Stat
          label="Abstained (best run)"
          value={fmt(best?.abstained)}
          sub={`unanswerable correct ${fmt(best?.correct_unanswerable)}`}
        />
        <Stat
          label="Judge vs. human"
          value={agreement ? `κ ${fmt(agreement.labels.correct?.kappa, 2)}` : "–"}
          sub={agreement ? `correct ${agreement.labels.correct?.agree}/${agreement.labels.correct?.n} · faithful κ ${fmt(agreement.labels.faithful?.kappa, 2)}` : "no labels yet"}
        />
      </div>

      <section className="card">
        <div className="card-head">
          <h2>Runs</h2>
          <label className="small">
            <input type="checkbox" checked={showDev} onChange={(e) => setShowDev(e.target.checked)} /> show dev-split runs
          </label>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>setup</th>
                <th>recall@10</th>
                <th>MRR@10</th>
                <th>correct</th>
                <th>unanswerable</th>
                <th>abstained</th>
                <th>faithful*</th>
                <th>cost (full)</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.folder} className={r === best ? "best" : ""}>
                  <td>
                    <code>{r.name}</code> {r === best && <span className="badge">best</span>}
                    {r.split === "dev" && <span className="badge dev">dev</span>}
                  </td>
                  <td className="small muted">{setupLabel(r)}</td>
                  <td>{ci(r.recall_at_10)}</td>
                  <td>{fmt(r.mrr_at_10)}</td>
                  <td>{ci(r.correct)}</td>
                  <td>{fmt(r.correct_unanswerable)}</td>
                  <td>{fmt(r.abstained)}</td>
                  <td>{fmt(r.faithful)}</td>
                  <td>{r.cost_eur_full === null ? "–" : `€${r.cost_eur_full.toFixed(3)}`}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="small muted">
          * faithful: judge_v2 agrees with human labels only at chance level (see "Judge vs. human"); read it as a
          rough trend. Cost (full): cache hits priced from stored usage, so runs are comparable; empty for runs
          logged before this was recorded.
        </div>
      </section>

      <div className="grid3">
        <section className="card">
          <h2>recall@10 and correct (Haiku pipeline)</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chart} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid stroke="#262a36" vertical={false} />
              <XAxis dataKey="name" tick={{ fill: "#9aa3b2", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#9aa3b2", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#151821", border: "1px solid #2a2f3d" }} />
              <Legend />
              <Bar dataKey="recall@10" fill="#6c7bd9">
                <ErrorBar dataKey="recallErr" stroke="#d6dbe6" width={4} />
              </Bar>
              <Bar dataKey="correct" fill="#3fb68b">
                <ErrorBar dataKey="correctErr" stroke="#d6dbe6" width={4} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="small muted">Whiskers = 95% CI.</div>
        </section>

        <section className="card">
          <h2>Text-layer vs. scanned documents</h2>
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>text R@10</th>
                <th>scanned R@10</th>
              </tr>
            </thead>
            <tbody>
              {ocrRuns.map((r) => (
                <tr key={r.folder}>
                  <td>
                    <code>{r.name}</code>
                  </td>
                  <td>{fmt(r.text_layer.recall_at_10)}</td>
                  <td>{fmt(r.scanned.recall_at_10)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="note">
            OCR only changes the scanned documents: text-layer recall stays flat, scanned recall rises from zero.
          </div>
        </section>

        <section className="card">
          <h2>Abstention experiment</h2>
          <table>
            <thead>
              <tr>
                <th>run</th>
                <th>correct</th>
                <th>abst. (answerable)</th>
                <th>unanswerable</th>
                <th>gate</th>
              </tr>
            </thead>
            <tbody>
              {abstention.map((r) => (
                <tr key={r.folder}>
                  <td>
                    <code>{r.setup.prompt_version}</code>
                  </td>
                  <td>{fmt(r.correct?.mean)}</td>
                  <td>{fmt(r.abstained_answerable)}</td>
                  <td>{fmt(r.correct_unanswerable)}</td>
                  <td>{fmt(r.gate_answerable)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="note">
            "gate" = share of answerable questions stopped by the rerank threshold before the LLM: the abstentions
            come from the LLM, so the fix was the prompt.
          </div>
        </section>
      </div>
    </div>
  );
}
