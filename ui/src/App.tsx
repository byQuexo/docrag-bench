import { lazy, Suspense, useEffect, useState } from "react";
import { Chat } from "./Chat";

// The dashboard pulls in the chart library; load it only when it's opened.
const Dashboard = lazy(() => import("./Dashboard").then((m) => ({ default: m.Dashboard })));

// Two views, switched by the URL hash (#chat / #dashboard): no router needed.
type View = "chat" | "dashboard";
const current = (): View => (location.hash === "#dashboard" ? "dashboard" : "chat");

export function App() {
  const [view, setView] = useState<View>(current);
  useEffect(() => {
    const onHash = () => setView(current());
    addEventListener("hashchange", onHash);
    return () => removeEventListener("hashchange", onHash);
  }, []);

  return (
    <div className="app">
      <header className="topbar">
        <span className="logo">d</span>
        <strong>docrag-bench</strong>
        <nav>
          <a href="#chat" className={view === "chat" ? "active" : ""}>
            Chat with sources
          </a>
          <a href="#dashboard" className={view === "dashboard" ? "active" : ""}>
            Benchmark dashboard
          </a>
        </nav>
      </header>
      {view === "chat" ? (
        <Chat />
      ) : (
        <Suspense fallback={<div className="page muted">Loading…</div>}>
          <Dashboard />
        </Suspense>
      )}
    </div>
  );
}
