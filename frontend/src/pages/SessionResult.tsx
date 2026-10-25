import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { deleteSession, getSession, type SessionOut } from "../api";

const BADGE: Record<string, string> = {
  ACCEPT: "bg-green-100 text-green-900 border-green-400",
  HUMAN_REVIEW: "bg-amber-100 text-amber-900 border-amber-400",
  REJECT: "bg-red-100 text-red-900 border-red-400",
  UNAVAILABLE: "bg-slate-100 text-slate-800 border-slate-400",
};

const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d);

export default function SessionResult() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const token = sessionStorage.getItem(`codmav-token-${id}`) ?? "";
  const [s, setS] = useState<SessionOut | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    async function poll() {
      try {
        const r = await getSession(id, token);
        if (stop) return;
        setS(r);
        if (["queued", "processing", "uploading"].includes(r.status)) setTimeout(poll, 1500);
      } catch (e) {
        setErr(String(e));
      }
    }
    poll();
    return () => {
      stop = true;
    };
  }, [id, token]);

  async function erase() {
    if (!confirm("Delete this session and all its data? This cannot be undone.")) return;
    await deleteSession(id, token);
    sessionStorage.removeItem(`codmav-token-${id}`);
    nav("/");
  }

  if (err) return <p role="alert" className="text-red-700">{err}</p>;
  if (!s) return <p>Loading…</p>;

  return (
    <section className="space-y-6" aria-live="polite">
      <div className="bg-white rounded-lg shadow p-6 space-y-2">
        <h1 className="text-xl font-semibold">Session {s.session_id.slice(0, 8)}</h1>
        <p>Status: <strong>{s.status}</strong></p>
        {s.error && <p role="alert" className="text-red-700">{s.error}</p>}
        {s.decision && (
          <p>
            Decision:{" "}
            <span className={`inline-block border rounded px-3 py-1 font-semibold ${BADGE[s.decision] ?? ""}`}>
              {s.decision.replace("_", " ")}
            </span>
          </p>
        )}
        {s.decision_reason && <p className="text-slate-700">{s.decision_reason}</p>}
        <p className="text-xs text-slate-500">Model bundle: {s.bundle_version ?? "—"}</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="font-semibold mb-3">Module outputs</h2>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left border-b">
              <th scope="col" className="py-1">Module</th><th scope="col">Status</th>
              <th scope="col">Score (1 = genuine)</th><th scope="col">Confidence</th><th scope="col">Note</th>
            </tr>
          </thead>
          <tbody>
            {s.modules.map((m) => (
              <tr key={m.module} className="border-b align-top">
                <td className="py-1">{m.module} · {m.title}</td>
                <td>
                  <span className={m.status === "placeholder" ? "bg-slate-200 px-2 rounded text-xs font-semibold" : ""}>
                    {m.status.toUpperCase()}
                  </span>
                </td>
                <td>{fmt(m.score)}</td>
                <td>{fmt(m.confidence)}</td>
                <td className="text-slate-600">{m.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {s.quality && (
        <div className="bg-white rounded-lg shadow p-6">
          <h2 className="font-semibold mb-3">Capture quality vector</h2>
          <dl className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-1 text-sm">
            {Object.entries(s.quality).map(([k, v]) => (
              <div key={k} className="flex justify-between border-b">
                <dt className="text-slate-600">{k}</dt><dd>{fmt(v as number | null, 3)}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      <button onClick={erase} className="px-4 py-2 rounded border border-red-700 text-red-700">
        Delete my data
      </button>
    </section>
  );
}
