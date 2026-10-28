import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createSession, submit, uploadDocument, uploadVideo, type CreatedSession, type PublicConfig } from "../api";
import Recorder from "../components/Recorder";

type Step = "consent" | "document" | "video" | "submitting";
const STEPS: { key: Step; label: string }[] = [
  { key: "consent", label: "Consent" },
  { key: "document", label: "ID document" },
  { key: "video", label: "Selfie video" },
  { key: "submitting", label: "Analysis" },
];

export default function KycFlow({ cfg }: { cfg: PublicConfig }) {
  const nav = useNavigate();
  const [step, setStep] = useState<Step>("consent");
  const [agree, setAgree] = useState(false);
  const [session, setSession] = useState<CreatedSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [docName, setDocName] = useState<string | null>(null);

  async function start() {
    setBusy(true);
    setError(null);
    try {
      const s = await createSession(cfg.consent_version);
      sessionStorage.setItem(`codmav-token-${s.session_id}`, s.token);
      setSession(s);
      setStep("document");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onDocument(f: File | undefined) {
    if (!f || !session) return;
    if (f.size > cfg.max_document_bytes) return setError("File too large.");
    setBusy(true);
    setError(null);
    try {
      await uploadDocument(session, f);
      setDocName(f.name);
      setStep("video");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onVideo(blob: Blob) {
    if (!session) return;
    if (blob.size > cfg.max_video_bytes) return setError("Video too large.");
    setStep("submitting");
    setError(null);
    try {
      await uploadVideo(session, blob);
      await submit(session);
      nav(`/session/${session.session_id}`);
    } catch (e) {
      setError(String(e));
      setStep("video");
    }
  }

  const idx = STEPS.findIndex((s) => s.key === step);

  return (
    <section aria-live="polite" className="space-y-6">
      <ol className="flex flex-wrap gap-2" aria-label="Progress">
        {STEPS.map((s, i) => (
          <li key={s.key} aria-current={i === idx ? "step" : undefined}
            className={`flex items-center gap-2 rounded-full px-4 py-1.5 text-sm ring-1 ${i < idx ? "bg-emerald-50 ring-emerald-200 text-emerald-800"
              : i === idx ? "bg-blue-600 ring-blue-600 text-white" : "bg-white ring-slate-200 text-slate-500"}`}>
            <span className="font-semibold">{i < idx ? "✓" : i + 1}</span> {s.label}
          </li>
        ))}
      </ol>

      {error && <p role="alert" className="card text-rose-700 py-3">{error}</p>}

      {step === "consent" && (
        <div className="card max-w-3xl space-y-4">
          <h1 className="text-2xl font-semibold">Verify your identity</h1>
          <p className="text-slate-600">We check an ID document and a short selfie video for fraud: deepfakes, screen or photo replays,
            liveness, and whether your face matches the ID photo.</p>
          <ul className="grid sm:grid-cols-2 gap-2 text-sm">
            {[`Uploads are stored encrypted and deleted after ${cfg.media_ttl_days} days`,
              "Derived measurements (quality statistics, scores) are kept for monitoring",
              "Facial images are biometric data",
              "You can delete everything from the result page"].map((t) => (
              <li key={t} className="rounded-xl bg-slate-50 ring-1 ring-slate-200 px-3 py-2">{t}</li>))}
          </ul>
          <p className="text-sm text-slate-500">Use only your own face and a <strong>synthetic or specimen ID</strong>, never a real person's
            ID document. Consent text version {cfg.consent_version}.</p>
          <label className="flex items-start gap-2">
            <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} className="mt-1 h-4 w-4" />
            <span>I consent to the processing described above.</span>
          </label>
          <button disabled={!agree || busy} onClick={start} className="btn-primary">Continue →</button>
        </div>
      )}

      {step === "document" && session && (
        <div className="card max-w-3xl space-y-4">
          <h1 className="text-2xl font-semibold">Upload your ID document</h1>
          <label htmlFor="doc"
            className="flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 p-10 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50/50">
            <span className="text-4xl" aria-hidden>🪪</span>
            <span className="font-medium">Click to choose a file</span>
            <span className="text-sm text-slate-500">JPEG, PNG, WebP or PDF (first page) · max {Math.round(cfg.max_document_bytes / 1048576)} MB</span>
          </label>
          <input id="doc" type="file" className="sr-only" accept="image/jpeg,image/png,image/webp,application/pdf" disabled={busy}
            onChange={(e) => onDocument(e.target.files?.[0])} />
          {busy && <p className="text-sm text-slate-600">Uploading…</p>}
          {cfg.demo_mode && (
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <button onClick={() => setStep("video")} className="btn-ghost">Skip: test the video only</button>
              <span className="text-slate-500">Demo mode: the document check and face match are then not run.</span>
            </div>
          )}
        </div>
      )}

      {step === "video" && session && (
        <div className="space-y-4">
          {docName && <p className="text-sm text-emerald-700">✓ ID document uploaded: {docName}</p>}
          <Recorder maxMs={session.duration_ms} onDone={onVideo} />
          {cfg.demo_mode && (
            <div className="card space-y-2 bg-amber-50/80 ring-amber-200">
              <label htmlFor="vidfile" className="block text-sm font-medium">
                Demo mode: or upload a video file (e.g. a dataset clip to test deepfake detection)
              </label>
              <input id="vidfile" type="file" accept="video/mp4,video/webm,video/quicktime" className="text-sm"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) onVideo(f); }} />
            </div>
          )}
        </div>
      )}

      {step === "submitting" && (
        <div className="card max-w-3xl flex items-center gap-4">
          <span className="h-8 w-8 rounded-full border-4 border-blue-200 border-t-blue-600 animate-spin" aria-hidden />
          <p>Uploading and starting the analysis…</p>
        </div>
      )}
    </section>
  );
}
