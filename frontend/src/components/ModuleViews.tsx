import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ModuleOut } from "../api";

const pct = (v: number | null) => (v === null ? "—" : `${Math.round(v * 100)}%`);

/** Horizontal score bar: 1 = genuine. Colour + text label (not colour alone). */
export function ScoreBar({ m }: { m: ModuleOut }) {
  if (m.status !== "ok" || m.score === null)
    return <span className="text-xs bg-slate-200 rounded px-2 py-0.5 font-semibold">{m.status.toUpperCase()}</span>;
  const s = m.score;
  const tone = s >= 0.7 ? "bg-green-600" : s >= 0.4 ? "bg-amber-500" : "bg-red-600";
  const word = s >= 0.7 ? "looks genuine" : s >= 0.4 ? "uncertain" : "looks fraudulent";
  return (
    <div className="w-full">
      <div className="flex justify-between text-xs"><span>{pct(s)} genuine · {word}</span>
        <span>confidence {pct(m.confidence)}</span></div>
      <div className="h-3 bg-slate-200 rounded" role="meter" aria-valuemin={0} aria-valuemax={100}
        aria-valuenow={Math.round(s * 100)} aria-label={`${m.title} genuineness`}>
        <div className={`h-3 rounded ${tone}`} style={{ width: `${s * 100}%`, opacity: 0.4 + 0.6 * (m.confidence ?? 0) }} />
      </div>
    </div>
  );
}

export function RppgPlot({ wf }: { wf: { t: number[]; pos: number[] } }) {
  const data = wf.t.map((t, i) => ({ t, v: wf.pos[i] }));
  return (
    <figure aria-label="Remote PPG waveform">
      <ResponsiveContainer width="100%" height={140}>
        <LineChart data={data}><XAxis dataKey="t" unit="s" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 10 }} />
          <YAxis hide /><Tooltip /><Line dataKey="v" dot={false} stroke="#dc2626" strokeWidth={1.5} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <figcaption className="text-xs text-slate-600">rPPG (POS) pulse signal from forehead/cheek skin colour.</figcaption>
    </figure>
  );
}

export function LandmarkOverlay({ ov }: { ov: { landmarks: number[][] | null; frame_shape: number[] } }) {
  if (!ov?.landmarks) return null;
  const [h, w] = ov.frame_shape;
  return (
    <figure>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full max-w-xs bg-slate-900 rounded" role="img"
        aria-label="Detected facial landmarks (points only, no image stored)">
        {ov.landmarks.map(([x, y], i) => <circle key={i} cx={x} cy={y} r={Math.max(w, h) / 400} fill="#38bdf8" />)}
      </svg>
      <figcaption className="text-xs text-slate-600">478 MediaPipe landmarks (points only).</figcaption>
    </figure>
  );
}

export function Img({ b64, alt }: { b64?: string | null; alt: string }) {
  if (!b64) return null;
  return <figure><img src={`data:image/png;base64,${b64}`} alt={alt} className="rounded max-w-xs" />
    <figcaption className="text-xs text-slate-600">{alt}</figcaption></figure>;
}

type Step = { kind: string; prompt: string; passed: boolean; [k: string]: unknown };
export function ChallengeSteps({ steps }: { steps: Step[] }) {
  return (
    <ul className="text-sm space-y-1">
      {steps.map((s, i) => (
        <li key={i}><span aria-hidden>{s.passed ? "✅" : "❌"}</span> <strong>{s.passed ? "Passed" : "Failed"}</strong>: {s.prompt}
          <span className="text-xs text-slate-500"> {Object.entries(s).filter(([k]) => !["kind", "prompt", "passed"].includes(k))
            .map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ")}</span></li>
      ))}
    </ul>
  );
}
