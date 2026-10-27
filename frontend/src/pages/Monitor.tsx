import { useCallback, useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

// Admin API: loopback-only in demo mode; otherwise an X-API-Key is required (kept in sessionStorage only).
const keyHeader = (): Record<string, string> => {
  const k = sessionStorage.getItem("codmav-admin-key");
  return k ? { "X-API-Key": k } : {};
};
async function api<T>(path: string, method = "GET"): Promise<T> {
  const r = await fetch(path, { method, headers: keyHeader() });
  if (!r.ok) throw new Error(`${r.status}: ${(await r.json().catch(() => ({}))).detail ?? r.statusText}`);
  return r.json();
}

type Status = { bundle_version: string; reference_available: boolean; reference_n: number; window: number | null;
  quality_names: string[]; logged_sessions: number; open_alerts: number; novelty_space: Record<string, unknown> | null };
type Feat = { feature: string; ks_q: number | null; psi: number | null; psi_crit: number | null; drifted: boolean; mean_shift_sd?: number };
type Report = { id: number; created_at: string; window_start: string; window_end: string; label: string;
  confidence: number | null; tests: Record<string, unknown>; features: Feat[] };
type Alert = { id: number; created_at: string; type: string; severity: string; confidence: number | null;
  bundle_version: string; status: string; recommendation: string | null; evidence: Record<string, any> };

const LABEL_STYLE: Record<string, string> = {
  NONE: "bg-green-100 text-green-900", OUTPUT_DRIFT: "bg-sky-100 text-sky-900", QUALITY_DRIFT: "bg-amber-100 text-amber-900",
  CONCEPT_DRIFT: "bg-orange-200 text-orange-900", NOVEL_ATTACK: "bg-red-200 text-red-900", MIXED: "bg-fuchsia-200 text-fuchsia-900",
};
const SEV: Record<string, string> = { low: "border-sky-400", medium: "border-amber-500", high: "border-orange-600", critical: "border-red-700" };

function heatColor(q: number | null): string {
  if (q === null) return "#e5e7eb";
  const s = Math.min(1, -Math.log10(Math.max(q, 1e-12)) / 6);          // 0 (q=1) .. 1 (q<=1e-6)
  return `rgba(220, 38, 38, ${0.08 + 0.92 * s})`;
}

export default function Monitor() {
  const [st, setSt] = useState<Status | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [detail, setDetail] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [win, setWin] = useState(200);
  const [alpha, setAlpha] = useState(0.05);
  const [key, setKey] = useState(sessionStorage.getItem("codmav-admin-key") ?? "");

  const load = useCallback(async () => {
    try {
      setErr(null);
      const s = await api<Status>("/api/monitor/status");
      setSt(s);
      setWin(s.window ?? 200);
      setReports(await api<Report[]>("/api/monitor/reports"));
      setAlerts(await api<Alert[]>("/api/monitor/alerts"));
    } catch (e) { setErr(String(e)); }
  }, []);
  useEffect(() => { load(); }, [load]);

  async function act(path: string) {
    setBusy(true); setErr(null);
    try {
      const r = await api<any>(path, "POST");
      if (r.report) setDetail(r.report);
      if (r.status === "insufficient_data") setErr(`Not enough sessions in the window (${r.n}/${r.needed}).`);
      await load();
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }
  async function openReport(id: number) {
    try { setDetail({ ...(await api<any>(`/api/monitor/reports/${id}`)), _id: id }); } catch (e) { setErr(String(e)); }
  }

  const names = st?.quality_names ?? [];
  const mv = detail?.multivariate ?? detail;
  const hist = mv?.evidence_histograms ?? {};
  const topFeats: Feat[] = (detail?.per_feature?.features ?? detail?.per_feature ?? [])
    .filter((f: Feat) => f.ks_q !== null).sort((a: Feat, b: Feat) => (Number(b.drifted) - Number(a.drifted)) || ((a.ks_q ?? 1) - (b.ks_q ?? 1))).slice(0, 5);

  return (
    <section className="space-y-6">
      <div className="bg-white rounded-lg shadow p-4 flex flex-wrap gap-4 items-end">
        <div>
          <h1 className="text-xl font-semibold">Dashboard 2: drift and novel-attack monitoring</h1>
          <p className="text-sm text-slate-600">Bundle {st?.bundle_version ?? "—"} · reference n={st?.reference_n ?? 0} ·
            logged sessions {st?.logged_sessions ?? 0} · open alerts {st?.open_alerts ?? 0}</p>
          {st && !st.reference_available && <p role="alert" className="text-red-700 text-sm">Active bundle has no monitoring reference.</p>}
        </div>
        <label className="text-sm">Window (sessions)
          <input type="number" min={50} max={1000} value={win} onChange={(e) => setWin(+e.target.value)} className="ml-2 border rounded px-2 w-24" /></label>
        <label className="text-sm">α
          <input type="number" step={0.01} min={0.001} max={0.2} value={alpha} onChange={(e) => setAlpha(+e.target.value)} className="ml-2 border rounded px-2 w-20" /></label>
        <button disabled={busy} onClick={() => act(`/api/monitor/run?window=${win}&alpha=${alpha}`)} className="px-3 py-1 rounded bg-blue-700 text-white disabled:opacity-50">Analyse latest window</button>
        <span className="text-sm text-slate-600">Simulate (held-out features, no media):</span>
        {(["stationary", "domain_drift", "unseen_generator"] as const).map((s) => (
          <button key={s} disabled={busy} onClick={() => act(`/api/monitor/simulate?scenario=${s}&n=${win}&seed=${Date.now() % 1000}`)}
            className="px-2 py-1 rounded border text-sm disabled:opacity-50">{s.replace("_", " ")}</button>))}
        <label className="text-sm ml-auto">Admin key
          <input type="password" value={key} onChange={(e) => setKey(e.target.value)} onBlur={() => { sessionStorage.setItem("codmav-admin-key", key); load(); }}
            className="ml-2 border rounded px-2 w-36" aria-label="Admin API key (not needed in demo mode)" /></label>
      </div>
      {err && <p role="alert" className="text-red-700">{err}</p>}

      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="font-semibold mb-2">Drift timeline</h2>
        {reports.length === 0 ? <p className="text-sm text-slate-600">No reports yet. Simulate a stream and analyse a window.</p> : (
          <ol className="flex flex-wrap gap-2" aria-label="Window labels over time">
            {reports.map((r) => (
              <li key={r.id}><button onClick={() => openReport(r.id)} className={`text-xs rounded px-2 py-1 ${LABEL_STYLE[r.label] ?? ""}`}
                title={`${r.window_start} → ${r.window_end}`}>#{r.id} {r.label.replace("_", " ")} {r.confidence !== null ? `(${(r.confidence * 100).toFixed(0)}%)` : ""}</button></li>))}
          </ol>)}
      </div>

      {reports.length > 0 && (
        <div className="bg-white rounded-lg shadow p-4 overflow-x-auto">
          <h2 className="font-semibold mb-2">Per-feature drift heatmap <span className="text-xs text-slate-500">(colour = −log10 BH q of KS vs subject-level null; outlined = drifted)</span></h2>
          <table className="text-xs border-collapse">
            <thead><tr><th scope="col" className="text-left pr-2">feature</th>{reports.map((r) => <th key={r.id} scope="col" className="px-1">#{r.id}</th>)}</tr></thead>
            <tbody>{names.map((n) => (
              <tr key={n}><th scope="row" className="text-left pr-2 font-normal">{n}</th>
                {reports.map((r) => { const f = r.features.find((x) => x.feature === n);
                  return <td key={r.id} title={f ? `q=${f.ks_q?.toExponential(1)} PSI=${f.psi?.toFixed(3)}` : "n/a"}
                    style={{ background: heatColor(f?.ks_q ?? null), outline: f?.drifted ? "2px solid #7f1d1d" : undefined }} className="w-8 h-5" />; })}
              </tr>))}</tbody>
          </table>
        </div>)}

      {detail && (
        <div className="bg-white rounded-lg shadow p-4 space-y-3">
          <h2 className="font-semibold">Window detail {detail._id ? `#${detail._id}` : ""}: <span className={`px-2 rounded ${LABEL_STYLE[detail.label] ?? ""}`}>{detail.label}</span></h2>
          {mv?.tests && <p className="text-sm">Tests: {Object.entries(mv.tests).map(([k, v]) => `${k}=${typeof v === "number" ? (v as number).toFixed(3) : String(v)}`).join(" · ")}</p>}
          <div>
            <h3 className="font-medium text-sm">Top drifting quality features</h3>
            <ul className="text-sm list-disc pl-5">{topFeats.map((f) => <li key={f.feature}>{f.feature}: q={f.ks_q?.toExponential(1)}, PSI {f.psi?.toFixed(3)} (crit {f.psi_crit?.toFixed(3)}){f.mean_shift_sd !== undefined ? `, shift ${f.mean_shift_sd.toFixed(2)} SD` : ""} {f.drifted ? "· DRIFTED" : ""}</li>)}</ul>
          </div>
          <div className="grid md:grid-cols-3 gap-4">
            {Object.entries(hist).map(([f, h]: [string, any]) => (
              <figure key={f}><figcaption className="text-xs font-medium">{f}: reference vs current</figcaption>
                <ResponsiveContainer width="100%" height={160}>
                  <BarChart data={h.ref.map((r: number, i: number) => ({ x: Number(h.edges[i]).toFixed(2), ref: r, cur: h.cur[i] }))}>
                    <CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" tick={{ fontSize: 9 }} /><YAxis tick={{ fontSize: 9 }} /><Tooltip /><Legend />
                    <Bar dataKey="ref" fill="#64748b" /><Bar dataKey="cur" fill="#dc2626" />
                  </BarChart></ResponsiveContainer></figure>))}
          </div>
          {mv?.novelty_excess && (
            <p className="text-sm">Novelty: {mv.novelty_excess.k_below_alpha}/{mv.novelty_excess.n} sessions below α (rate {(mv.novelty_excess.rate * 100).toFixed(1)}%,
              null mean {(mv.novelty_excess.expected * 100).toFixed(1)}%, p={Number(mv.novelty_excess.p_value).toFixed(3)}).</p>)}
          {mv?.clusters && (
            <div className="text-sm"><h3 className="font-medium">Novel-candidate clusters</h3>
              <p>{mv.clusters.n_outliers} outliers · coherent: {String(mv.clusters.coherent)} · silhouette {mv.clusters.silhouette ?? "n/a"} ·
                nearest known classes {JSON.stringify(mv.clusters.outlier_nearest_class)}</p>
              <ul className="list-disc pl-5">{(mv.clusters.clusters ?? []).map((c: any) => <li key={c.cluster}>cluster {c.cluster}: size {c.size}, cohesion {c.cohesion.toFixed(3)}</li>)}</ul>
              {mv.cluster_members?.length > 0 && <p className="text-xs text-slate-500">Members (session ids, no media stored for simulated sessions): {mv.cluster_members.slice(0, 10).join(", ")}…</p>}
            </div>)}
          {detail.recommendations?.length > 0 && (
            <div><h3 className="font-medium text-sm">Recommended actions</h3>
              <ul className="text-sm list-disc pl-5">{detail.recommendations.map((r: any, i: number) => <li key={i}>{r.message}</li>)}</ul></div>)}
        </div>)}

      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="font-semibold mb-2">Alert feed</h2>
        {alerts.length === 0 ? <p className="text-sm text-slate-600">No alerts.</p> : (
          <ul className="space-y-2">{alerts.map((a) => (
            <li key={a.id} className={`border-l-4 pl-3 ${SEV[a.severity] ?? ""}`}>
              <p className="text-sm"><strong>{a.type.replace("_", " ").toUpperCase()}</strong> · {a.severity} · confidence {a.confidence !== null ? (a.confidence * 100).toFixed(0) + "%" : "—"} ·
                {new Date(a.created_at).toLocaleString()} · {a.evidence?.window} · bundle {a.bundle_version} · {a.status}</p>
              {a.recommendation && <p className="text-sm text-slate-700 whitespace-pre-line">{a.recommendation}</p>}
              {a.status === "open" && <button onClick={() => act(`/api/monitor/alerts/${a.id}/ack`)} className="text-xs underline">acknowledge</button>}
            </li>))}</ul>)}
      </div>
    </section>
  );
}
