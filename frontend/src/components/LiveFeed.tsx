import { Fragment, useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

type Unusual = { feature: string; p_two_sided: number; direction: string; value: number; reference_median: number };
type Row = {
  session_id: string; created_at: string; status: string; source: string; capture?: string | null;
  decision: string | null; fused_score: number | null; decision_reason?: string | null;
  scores?: Record<string, number | null>; liveness_actions?: { done: number; total: number };
  capture_check?: { status: string; unusual_features: Unusual[] }; artifact_novelty_p?: number;
};
type Feed = { sessions: Row[]; n_completed: number; window: { size: number; min_sessions: number; ready: boolean };
  reference_available: boolean; note: string };

const DEC: Record<string, string> = {
  ACCEPT: "bg-emerald-100 text-emerald-800", HUMAN_REVIEW: "bg-amber-100 text-amber-800",
  REJECT: "bg-rose-100 text-rose-800", UNAVAILABLE: "bg-slate-100 text-slate-700",
};

function Score({ v }: { v: number | null | undefined }) {
  if (v === null || v === undefined) return <span className="text-slate-300">—</span>;
  const c = v >= 0.6 ? "text-emerald-700" : v < 0.4 ? "text-rose-700 font-semibold" : "text-amber-700";
  return <span className={`tabular-nums ${c}`}>{Math.round(v * 100)}</span>;
}

function verdict(r: Row): { t: string; c: string } {
  const m2 = r.scores?.M2, m3 = r.scores?.M3;
  if (m2 === null || m2 === undefined) return { t: "no face", c: "bg-slate-100 text-slate-600" };
  if (m2 < 0.4) return { t: "DEEPFAKE", c: "bg-rose-600 text-white" };
  if (m3 !== null && m3 !== undefined && m3 < 0.4) return { t: "SPOOF", c: "bg-rose-600 text-white" };
  if (m2 < 0.6 || (m3 !== null && m3 !== undefined && m3 < 0.6)) return { t: "UNCERTAIN", c: "bg-amber-400 text-amber-950" };
  return { t: "REAL", c: "bg-emerald-600 text-white" };
}

/** Live feed of analysed KYC sessions with per-session capture / novelty checks (auto-refresh). */
export default function LiveFeed({ api }: { api: <T>(path: string) => Promise<T> }) {
  const [source, setSource] = useState<"live" | "simulated" | "all">("live");
  const [feed, setFeed] = useState<Feed | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [auto, setAuto] = useState(true);
  const [open, setOpen] = useState<string | null>(null);
  const [updated, setUpdated] = useState<Date | null>(null);

  const load = useCallback(async () => {
    try {
      setFeed(await api<Feed>(`/api/monitor/sessions?source=${source}&limit=50`));
      setUpdated(new Date());
      setErr(null);
    } catch (e) { setErr(String(e)); }
  }, [api, source]);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (!auto) return;
    const iv = setInterval(load, 4000);
    return () => clearInterval(iv);
  }, [auto, load]);

  const rows = feed?.sessions ?? [];
  const busy = rows.some((r) => r.status !== "completed" && r.status !== "failed");
  const nDone = rows.filter((r) => r.status === "completed");
  const nFake = nDone.filter((r) => ["DEEPFAKE", "SPOOF"].includes(verdict(r).t)).length;
  const nUnusual = nDone.filter((r) => r.capture_check?.status === "unusual").length;

  return (
    <div className="card space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span className={`h-2.5 w-2.5 rounded-full ${auto ? "bg-emerald-500 animate-pulse" : "bg-slate-300"}`} aria-hidden />
          <h2 className="text-lg font-semibold">Live KYC sessions</h2>
        </div>
        <div role="tablist" aria-label="Session source" className="inline-flex gap-1 rounded-xl bg-slate-100 p-1">
          {(["live", "simulated", "all"] as const).map((s) => (
            <button key={s} role="tab" aria-selected={source === s} onClick={() => setSource(s)}
              className={`px-3 py-1 rounded-lg text-sm ${source === s ? "bg-white shadow font-medium" : "text-slate-600"}`}>{s}</button>))}
        </div>
        <label className="text-sm flex items-center gap-1"><input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} /> auto-refresh</label>
        <button onClick={load} className="btn-ghost !px-3 !py-1 text-sm">↻ Refresh</button>
        <span className="text-xs text-slate-500 ml-auto">{updated ? `updated ${updated.toLocaleTimeString()}` : ""}</span>
      </div>
      {err && <p role="alert" className="text-rose-700 text-sm">{err}</p>}

      {feed && (
        <div className="grid sm:grid-cols-4 gap-3 text-sm">
          <div className="rounded-xl bg-slate-50 ring-1 ring-slate-200 p-3"><p className="text-slate-500">Analysed ({source})</p>
            <p className="text-2xl font-bold">{feed.n_completed}</p></div>
          <div className="rounded-xl bg-rose-50 ring-1 ring-rose-200 p-3"><p className="text-rose-700">Deepfake / spoof (last {nDone.length})</p>
            <p className="text-2xl font-bold text-rose-800">{nFake}</p></div>
          <div className="rounded-xl bg-amber-50 ring-1 ring-amber-200 p-3"><p className="text-amber-700">Unusual capture vs reference</p>
            <p className="text-2xl font-bold text-amber-800">{nUnusual}</p></div>
          <div className={`rounded-xl ring-1 p-3 ${feed.window.ready ? "bg-emerald-50 ring-emerald-200" : "bg-slate-50 ring-slate-200"}`}>
            <p className="text-slate-600">Window drift analysis</p>
            <p className="font-semibold">{feed.window.ready ? "ready: click Analyse below" : `${feed.n_completed}/${feed.window.min_sessions} sessions needed`}</p>
            {!feed.window.ready && (
              <div className="h-1.5 mt-2 rounded-full bg-slate-200 overflow-hidden">
                <div className="h-full bg-blue-500" style={{ width: `${Math.min(100, (100 * feed.n_completed) / feed.window.min_sessions)}%` }} />
              </div>)}
          </div>
        </div>
      )}

      {rows.length === 0 ? (
        <p className="text-sm text-slate-500">No {source === "all" ? "" : source} sessions yet.
          {source === "live" && <> Run a verification on the <Link to="/" className="text-blue-700 underline">KYC page</Link>; it appears here as soon as it is analysed.</>}</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wide text-slate-500 border-b">
                <th className="py-2 pr-3">Time</th><th className="pr-3">Video</th><th className="pr-3">Decision</th>
                <th className="pr-3" title="fused P(genuine) ×100">Fused</th>
                {["M1", "M2", "M3", "M4", "M5"].map((m) => <th key={m} className="pr-2" title="P(genuine) ×100">{m}</th>)}
                <th className="pr-3">Actions</th><th className="pr-3">Capture vs reference</th>
                <th className="pr-3" title="conformal p-value of the artifact vector under the reference; small = unlike any known class">Novelty p</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => {
                const v = verdict(r);
                const pending = r.status !== "completed";
                const tok = sessionStorage.getItem(`codmav-token-${r.session_id}`);
                return (
                  <Fragment key={r.session_id}>
                    <tr onClick={() => setOpen(open === r.session_id ? null : r.session_id)}
                      className="border-b border-slate-100 hover:bg-slate-50 cursor-pointer">
                      <td className="py-2 pr-3 whitespace-nowrap tabular-nums">{new Date(r.created_at + (r.created_at.endsWith("Z") ? "" : "Z")).toLocaleTimeString()}</td>
                      <td className="pr-3">{pending ? <span className="text-xs text-blue-700">{r.status}…</span>
                        : r.status === "failed" ? <span className="text-xs text-rose-700">failed</span>
                        : <span className={`rounded-md px-2 py-0.5 text-xs font-bold ${v.c}`}>{v.t}</span>}</td>
                      <td className="pr-3">{r.decision && <span className={`rounded-md px-2 py-0.5 text-xs ${DEC[r.decision] ?? ""}`}>{r.decision.replace("_", " ")}</span>}</td>
                      <td className="pr-3"><Score v={r.fused_score} /></td>
                      {["M1", "M2", "M3", "M4", "M5"].map((m) => <td key={m} className="pr-2"><Score v={r.scores?.[m]} /></td>)}
                      <td className="pr-3 tabular-nums">{r.liveness_actions ? `${r.liveness_actions.done}/${r.liveness_actions.total}` : "—"}</td>
                      <td className="pr-3">{r.capture_check ? (r.capture_check.status === "typical"
                        ? <span className="text-emerald-700">typical</span>
                        : <span className="text-amber-700">unusual: {r.capture_check.unusual_features.slice(0, 2).map((f) => f.feature).join(", ")}
                          {r.capture_check.unusual_features.length > 2 && ` +${r.capture_check.unusual_features.length - 2}`}</span>) : "—"}</td>
                      <td className={`pr-3 tabular-nums ${(r.artifact_novelty_p ?? 1) < 0.05 ? "text-rose-700 font-semibold" : ""}`}>
                        {r.artifact_novelty_p !== undefined ? r.artifact_novelty_p.toFixed(3) : "—"}</td>
                      <td className="text-xs text-slate-500">{r.source}{r.capture ? ` · ${r.capture}` : ""}</td>
                    </tr>
                    {open === r.session_id && (
                      <tr className="bg-slate-50">
                        <td colSpan={14} className="p-3 text-xs space-y-1">
                          <p><strong>Session</strong> {r.session_id} {tok && <Link to={`/session/${r.session_id}`} className="text-blue-700 underline ml-2">open result page</Link>}</p>
                          {r.decision_reason && <p><strong>Reason:</strong> {r.decision_reason}</p>}
                          {r.capture_check?.unusual_features.length ? (
                            <ul className="list-disc pl-5">{r.capture_check.unusual_features.map((f) => (
                              <li key={f.feature}>{f.feature}: {f.value.toFixed(3)} ({f.direction}; reference median {f.reference_median.toFixed(3)}, two-sided p {f.p_two_sided})</li>))}</ul>
                          ) : <p>All capture-quality features are within the central 98% of the reference.</p>}
                          <p className="text-slate-500">{feed?.note}</p>
                        </td>
                      </tr>)}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
          {busy && <p className="text-xs text-blue-700 mt-2">A session is being analysed; this table refreshes automatically.</p>}
        </div>
      )}
    </div>
  );
}
