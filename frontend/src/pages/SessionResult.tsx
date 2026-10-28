import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { deleteSession, getSession, type ModuleOut, type SessionOut } from "../api";
import { Img, LandmarkOverlay, RppgPlot, ScoreBar } from "../components/ModuleViews";

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`);
const fmt = (v: number | null | undefined, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d);

type Tone = "good" | "bad" | "warn" | "na";
const TONE: Record<Tone, string> = {
  good: "bg-emerald-50 ring-emerald-200 text-emerald-900",
  bad: "bg-rose-50 ring-rose-200 text-rose-900",
  warn: "bg-amber-50 ring-amber-200 text-amber-900",
  na: "bg-slate-50 ring-slate-200 text-slate-600",
};
const BAR: Record<Tone, string> = { good: "bg-emerald-500", bad: "bg-rose-500", warn: "bg-amber-500", na: "bg-slate-300" };

/** Three-way reading of a calibrated P(genuine): <0.4 fraud, 0.4-0.6 uncertain, >=0.6 genuine. */
function read(m: ModuleOut | undefined): Tone {
  if (!m || m.status !== "ok" || m.score === null) return "na";
  return m.score >= 0.6 ? "good" : m.score < 0.4 ? "bad" : "warn";
}

type Frame = { t_s: number | null; p_real: number; thumb_png: string | null };
type Action = { action: string; label: string; done: boolean; at_s?: number | null; max_shift?: number; count?: number; note?: string };
type Actions = { actions: Action[]; all_done: boolean; duration_s: number; method: string;
  series: { t_s?: number[]; horizontal?: (number | null)[]; vertical?: (number | null)[] } };

const DECISION: Record<string, { label: string; cls: string }> = {
  ACCEPT: { label: "KYC ACCEPTED", cls: "bg-emerald-600" },
  HUMAN_REVIEW: { label: "SENT TO HUMAN REVIEW", cls: "bg-amber-500" },
  REJECT: { label: "KYC REJECTED", cls: "bg-rose-600" },
  UNAVAILABLE: { label: "NO DECISION", cls: "bg-slate-500" },
};

export default function SessionResult() {
  const { id = "" } = useParams();
  const nav = useNavigate();
  const token = sessionStorage.getItem(`codmav-token-${id}`) ?? "";
  const [s, setS] = useState<SessionOut | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [t0] = useState(() => Date.now());
  const [, tick] = useState(0);

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
    const iv = setInterval(() => tick((x) => x + 1), 1000);
    return () => {
      stop = true;
      clearInterval(iv);
    };
  }, [id, token]);

  async function erase() {
    if (!confirm("Delete this session and all its data? This cannot be undone.")) return;
    await deleteSession(id, token);
    sessionStorage.removeItem(`codmav-token-${id}`);
    nav("/");
  }

  if (err) return <p role="alert" className="card text-rose-700">{err}</p>;
  if (!s) return <p className="card">Loading…</p>;

  if (["queued", "processing", "uploading"].includes(s.status)) {
    const secs = Math.round((Date.now() - t0) / 1000);
    return (
      <div className="card max-w-2xl mx-auto text-center space-y-4 py-12" aria-live="polite">
        <span className="inline-block h-14 w-14 rounded-full border-[6px] border-blue-100 border-t-blue-600 animate-spin" aria-hidden />
        <h1 className="text-xl font-semibold">Analysing your video…</h1>
        <p className="text-slate-600 text-sm">Splitting the video into frames, tracking the face, and running the deepfake, replay,
          liveness, face-match and document checks. This usually takes 15–40 s on a laptop CPU.</p>
        <p className="text-slate-500 text-sm tabular-nums">{secs} s</p>
      </div>
    );
  }

  const by = Object.fromEntries(s.modules.map((m) => [m.module, m])) as Record<string, ModuleOut>;
  const m2 = by.M2, m3 = by.M3, m4 = by.M4, m5 = by.M5, m1 = by.M1;
  const e2 = (m2?.explanations ?? {}) as Record<string, any>;
  const e4 = (m4?.explanations ?? {}) as Record<string, any>;
  const frames: Frame[] = e2.frames ?? [];
  const actions: Actions | undefined = e4.actions;

  // ---------------- headline verdict (the video: real person vs deepfake / replay)
  const df = read(m2), rp = read(m3);
  let verdict = { title: "REAL PERSON", sub: "No deepfake or replay detected in the video.", tone: "good" as Tone, icon: "✓" };
  if (df === "na") verdict = { title: "NO FACE ANALYSED", sub: m2?.message ?? "The deepfake check could not run.", tone: "na", icon: "?" };
  else if (df === "bad") verdict = { title: "DEEPFAKE DETECTED", sub: "The face in the video looks synthetically manipulated.", tone: "bad", icon: "✕" };
  else if (rp === "bad") verdict = { title: "SPOOF SUSPECTED", sub: "The video looks like a replay of a screen or a printed photo.", tone: "bad", icon: "✕" };
  else if (df === "warn" || rp === "warn") verdict = { title: "UNCERTAIN", sub: "The video checks are not conclusive.", tone: "warn", icon: "!" };
  const hero: Record<Tone, string> = {
    good: "from-emerald-500 to-teal-600", bad: "from-rose-500 to-red-700", warn: "from-amber-400 to-orange-500", na: "from-slate-400 to-slate-600",
  };
  const dec = DECISION[s.decision ?? "UNAVAILABLE"] ?? DECISION.UNAVAILABLE;

  const checks: { key: string; title: string; m?: ModuleOut; good: string; bad: string; note?: string }[] = [
    { key: "M2", title: "Deepfake", m: m2, good: "Real face", bad: "Deepfake" },
    { key: "M3", title: "Replay / spoof", m: m3, good: "Live capture", bad: "Screen or photo replay" },
    { key: "M4", title: "Liveness signals", m: m4, good: "Live person", bad: "Liveness not confirmed" },
    { key: "M5", title: "Face vs ID photo", m: m5, good: "Same person", bad: "Different person" },
    { key: "M1", title: "ID document", m: m1, good: "Document looks genuine", bad: "Document looks altered",
      note: "Weak on ID designs not seen in training" },
  ];

  return (
    <section className="space-y-6" aria-live="polite">
      {s.status === "failed" && <p role="alert" className="card text-rose-700">Processing failed: {s.error}</p>}

      {/* ---------------- verdict */}
      <div className={`rounded-3xl bg-gradient-to-br ${hero[verdict.tone]} text-white p-8 shadow-lg`}>
        <div className="flex flex-wrap items-center gap-6">
          <div className="h-20 w-20 rounded-full bg-white/20 grid place-items-center text-5xl font-bold" aria-hidden>{verdict.icon}</div>
          <div className="flex-1 min-w-[240px]">
            <p className="text-sm uppercase tracking-widest text-white/80">Video verdict</p>
            <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight">{verdict.title}</h1>
            <p className="mt-1 text-white/90">{verdict.sub}</p>
          </div>
          {m2?.score !== null && m2?.score !== undefined && (
            <div className="text-right">
              <p className="text-sm text-white/80">Probability the face is real</p>
              <p className="text-5xl font-bold tabular-nums">{pct(m2.score)}</p>
              <p className="text-xs text-white/80">deepfake model confidence {pct(m2.confidence)}</p>
            </div>
          )}
        </div>
        <div className="mt-6 flex flex-wrap items-center gap-3 bg-black/15 rounded-2xl px-4 py-3">
          <span className={`rounded-lg px-3 py-1 text-sm font-bold ${dec.cls}`}>{dec.label}</span>
          <span className="text-sm text-white/95">{s.decision_reason}</span>
          {s.fused_score !== null && <span className="text-xs text-white/75 ml-auto">overall fused score {fmt(s.fused_score, 3)}</span>}
        </div>
      </div>

      {/* ---------------- check cards */}
      <div className="grid sm:grid-cols-2 lg:grid-cols-5 gap-3">
        {checks.map((c) => {
          const t = read(c.m);
          const word = t === "na" ? (c.m?.status === "not_applicable" ? "Not run" : "Unavailable") : t === "good" ? c.good : t === "bad" ? c.bad : "Uncertain";
          return (
            <div key={c.key} className={`rounded-2xl ring-1 p-4 ${TONE[t]}`}>
              <p className="text-xs font-medium uppercase tracking-wide opacity-70">{c.title}</p>
              <p className="text-lg font-bold leading-tight mt-1">{word}</p>
              {c.m?.score !== null && c.m?.score !== undefined ? (
                <>
                  <div className="h-2 mt-3 rounded-full bg-white/70 overflow-hidden"><div className={`h-full ${BAR[t]}`} style={{ width: `${c.m.score * 100}%` }} /></div>
                  <p className="text-xs mt-1 opacity-80">{pct(c.m.score)} genuine · conf {pct(c.m.confidence)}</p>
                </>
              ) : <p className="text-xs mt-3 opacity-80">{c.m?.message ?? "—"}</p>}
              {c.note && t !== "na" && <p className="text-[11px] mt-1 opacity-70">{c.note}</p>}
            </div>
          );
        })}
      </div>

      {/* ---------------- frame-by-frame */}
      {frames.length > 0 && (
        <div className="card space-y-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-lg font-semibold">Frame-by-frame deepfake analysis</h2>
            <p className="text-xs text-slate-500">{frames.length} most frontal frames from the whole video · green = looks real, red = looks fake</p>
          </div>
          <ol className="grid grid-cols-4 sm:grid-cols-8 gap-2">
            {frames.map((f, i) => {
              const t: Tone = f.p_real >= 0.6 ? "good" : f.p_real < 0.4 ? "bad" : "warn";
              return (
                <li key={i} className={`rounded-xl overflow-hidden ring-2 ${t === "good" ? "ring-emerald-400" : t === "bad" ? "ring-rose-500" : "ring-amber-400"}`}>
                  {f.thumb_png ? <img src={`data:image/png;base64,${f.thumb_png}`} alt={`Face at ${f.t_s ?? "?"} s`} className="w-full aspect-square object-cover" />
                    : <div className="aspect-square bg-slate-200" />}
                  <div className={`text-center text-xs font-semibold py-1 text-white ${BAR[t]}`}>{pct(f.p_real)} real</div>
                  <div className="text-center text-[10px] text-slate-500 py-0.5">{f.t_s !== null ? `${f.t_s.toFixed(1)} s` : ""}</div>
                </li>
              );
            })}
          </ol>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={frames.map((f) => ({ t: f.t_s, p: Math.round(f.p_real * 100) }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="t" type="number" unit=" s" domain={["dataMin", "dataMax"]} tick={{ fontSize: 11 }} />
              <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v) => [`${v}% real`, "frame"]} labelFormatter={(l) => `${l} s`} />
              <ReferenceLine y={50} stroke="#94a3b8" strokeDasharray="4 4" />
              <Line dataKey="p" stroke="#4f46e5" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="text-xs text-slate-500">The video verdict uses all frames together (more reliable). Single-frame scores are indicative:
            the model was trained on 16-frame averages, so individual frames are noisier.</p>
        </div>
      )}

      {/* ---------------- liveness actions */}
      {actions && (
        <div className="card space-y-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-lg font-semibold">Liveness actions found in the video</h2>
            <span className={`text-sm font-semibold ${actions.all_done ? "text-emerald-700" : "text-amber-700"}`}>
              {actions.actions.filter((a) => a.done).length}/{actions.actions.length} detected · {fmt(actions.duration_s, 1)} s analysed
            </span>
          </div>
          <ul className="grid sm:grid-cols-5 gap-2">
            {actions.actions.map((a) => (
              <li key={a.action} className={`rounded-xl ring-1 px-3 py-2 ${a.done ? TONE.good : TONE.warn}`}>
                <p className="font-semibold">{a.done ? "✓" : "✕"} {a.label}</p>
                <p className="text-xs opacity-80">
                  {a.action === "blink" ? `${a.count ?? 0} blink(s)` : a.done ? `at ${a.at_s ?? "?"} s` : "not detected"}
                  {a.max_shift !== undefined && ` · peak ${fmt(a.max_shift, 2)}`}
                </p>
              </li>
            ))}
          </ul>
          {actions.series?.t_s && (
            <ResponsiveContainer width="100%" height={180}>
              <LineChart data={actions.series.t_s.map((t, i) => ({ t, h: actions.series.horizontal?.[i], v: actions.series.vertical?.[i] }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="t" type="number" unit=" s" domain={["dataMin", "dataMax"]} tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Legend />
                <Line name="left (+) / right (−)" dataKey="h" stroke="#2563eb" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
                <Line name="down (+) / up (−)" dataKey="v" stroke="#db2777" dot={false} strokeWidth={2} isAnimationActive={false} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          )}
          <p className="text-xs text-slate-500">{actions.method}. Head pose is relative to your neutral pose. The thresholds are heuristic
            and not validated on a labelled liveness dataset; missing actions send the session to human review rather than rejecting it.</p>
        </div>
      )}

      {/* ---------------- technical details */}
      <details className="card">
        <summary className="cursor-pointer font-semibold">Technical details (all module outputs, fusion, capture quality)</summary>
        <div className="mt-4 space-y-6">
          {s.fused_score !== null && (
            <p className="text-sm text-slate-700">Fused P(genuine) {fmt(s.fused_score, 4)}
              {s.fusion_weights?.thresholds && <> · accept ≥ {fmt(s.fusion_weights.thresholds.t_high, 4)} · reject &lt; {fmt(s.fusion_weights.thresholds.t_low, 4)}</>}
              {s.fusion_strategy && <> · fusion {s.fusion_strategy}</>}{s.veto_module && <> · veto by {s.veto_module}</>}
              <span className="text-slate-500"> · bundle {s.bundle_version}</span></p>
          )}
          <ul className="space-y-5">
            {s.modules.map((m) => {
              const e = (m.explanations ?? {}) as Record<string, any>;
              return (
                <li key={m.module} className="border-b border-slate-100 pb-4">
                  <div className="flex items-baseline justify-between gap-4">
                    <h3 className="font-medium">{m.module} · {m.title}</h3>
                    <span className="text-xs text-slate-500">{m.model_version} · {Math.round(m.latency_ms)} ms</span>
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
                  {e.top_features && (
                    <p className="text-xs text-slate-600 mt-1">Top features: {e.top_features
                      .map((f: any) => `${f.feature} (${f.contribution > 0 ? "+" : ""}${f.contribution})`).join(", ")}</p>
                  )}
                </li>
              );
            })}
          </ul>
          {s.quality && (
            <div>
              <h3 className="font-medium mb-2">Capture quality vector</h3>
              <dl className="grid grid-cols-2 md:grid-cols-3 gap-x-6 gap-y-1 text-sm">
                {Object.entries(s.quality).map(([k, v]) => (
                  <div key={k} className="flex justify-between border-b border-slate-100">
                    <dt className="text-slate-600">{k}</dt><dd className="tabular-nums">{fmt(v as number | null, 3)}</dd>
                  </div>
                ))}
              </dl>
            </div>
          )}
        </div>
      </details>

      <div className="flex flex-wrap gap-3">
        <Link to="/" className="btn-primary">Start a new verification</Link>
        <button onClick={erase} className="btn-ghost text-rose-700">Delete my data</button>
      </div>
    </section>
  );
}
