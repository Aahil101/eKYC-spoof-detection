import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { deleteSession, getSession, type SessionOut } from "../api";
import { ChallengeSteps, Img, LandmarkOverlay, RppgPlot, ScoreBar } from "../components/ModuleViews";

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
        {s.fused_score !== null && (
          <div className="text-sm text-slate-700 space-y-1">
            <p>Fused score (P genuine): <strong>{fmt(s.fused_score, 4)}</strong>
              {s.fusion_weights?.thresholds && <> · accept ≥ {fmt(s.fusion_weights.thresholds.t_high, 4)} · reject &lt; {fmt(s.fusion_weights.thresholds.t_low, 4)}</>}
              {s.fusion_strategy && <> · fusion {s.fusion_strategy}</>}</p>
            {s.fusion_weights?.per_module && (
              <p>Module weights used: {Object.entries(s.fusion_weights.per_module).map(([m, w]) =>
                `${m} ${w === null ? "n/a" : (w as number).toFixed(2)}`).join(" · ")}
                {s.fusion_strategy === "F3" && <span className="text-xs text-slate-500"> (gradient-boosted stacking: no per-session linear weights; see top features)</span>}</p>
            )}
            {s.veto_module && <p className="text-red-700">Veto by {s.veto_module}.</p>}
          </div>
        )}
        <p className="text-xs text-slate-500">Model bundle: {s.bundle_version ?? "—"}</p>
      </div>

      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="font-semibold mb-3">Module outputs</h2>
        <ul className="space-y-4">
          {s.modules.map((m) => {
            const e = (m.explanations ?? {}) as Record<string, any>;
            return (
              <li key={m.module} className="border-b pb-3">
                <div className="flex items-baseline justify-between gap-4">
                  <h3 className="font-medium">{m.module} · {m.title}</h3>
                  <span className="text-xs text-slate-500">{m.model_version}</span>
                </div>
                <ScoreBar m={m} />
                {m.message && <p className="text-sm text-slate-600 mt-1">{m.message}</p>}
                <div className="flex flex-wrap gap-4 mt-2">
                  <Img b64={e.gradcam_png} alt="Grad-CAM: regions pushing towards 'fake'" />
                  <Img b64={e.ela_heatmap_png} alt="Error level analysis (q=90)" />
                  <Img b64={e.noise_heatmap_png} alt="Block noise-residual map" />
                  <Img b64={e.id_portrait_png} alt="Portrait found on the ID document" />
                  {e.overlay && <LandmarkOverlay ov={e.overlay} />}
                </div>
                {e.rppg_waveform && <RppgPlot wf={e.rppg_waveform} />}
                {e.challenge_steps && <ChallengeSteps steps={e.challenge_steps} />}
                {e.top_features && (
                  <p className="text-xs text-slate-600 mt-1">Top features: {e.top_features
                    .map((f: any) => `${f.feature} (${f.contribution > 0 ? "+" : ""}${f.contribution})`).join(", ")}</p>
                )}
              </li>
            );
          })}
        </ul>
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
