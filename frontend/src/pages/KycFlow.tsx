import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { createSession, submit, uploadDocument, uploadVideo, type CreatedSession, type PublicConfig } from "../api";
import Recorder from "../components/Recorder";

type Step = "consent" | "document" | "video" | "submitting";

export default function KycFlow({ cfg }: { cfg: PublicConfig }) {
  const nav = useNavigate();
  const [step, setStep] = useState<Step>("consent");
  const [agree, setAgree] = useState(false);
  const [session, setSession] = useState<CreatedSession | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

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
      setStep("video");
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onVideo(blob: Blob) {
    if (!session) return;
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

  return (
    <section aria-live="polite" className="space-y-6">
      <ol className="flex gap-4 text-sm" aria-label="Progress">
        {["consent", "document", "video"].map((s, i) => (
          <li key={s} className={step === s ? "font-semibold text-blue-700" : "text-slate-500"}>
            {i + 1}. {s === "consent" ? "Consent" : s === "document" ? "ID document" : "Selfie video"}
          </li>
        ))}
      </ol>

      {error && <p role="alert" className="text-red-700">{error}</p>}

      {step === "consent" && (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
          <h1 className="text-xl font-semibold">Before you start</h1>
          <p>This research prototype checks an ID document and a short selfie video for fraud. It will:</p>
          <ul className="list-disc pl-6 space-y-1">
            <li>store your uploads <strong>encrypted</strong> and delete them after {cfg.media_ttl_days} days;</li>
            <li>keep derived measurements (image-quality statistics, model scores) for research monitoring;</li>
            <li>process facial images, which are biometric data;</li>
            <li>let you delete everything at any time from the result page.</li>
          </ul>
          <p className="text-sm text-slate-600">
            Use only your own face and a <strong>synthetic or specimen ID</strong>. Never upload a real person's ID
            document. Consent text version {cfg.consent_version}. Details: docs/PRIVACY.md.
          </p>
          <label className="flex items-start gap-2">
            <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} className="mt-1" />
            <span>I consent to the processing described above.</span>
          </label>
          <button disabled={!agree || busy} onClick={start}
            className="px-4 py-2 rounded bg-blue-700 text-white disabled:opacity-50">Continue</button>
        </div>
      )}

      {step === "document" && (
        <div className="bg-white rounded-lg shadow p-6 space-y-4">
          <h1 className="text-xl font-semibold">Upload ID document</h1>
          <label htmlFor="doc" className="block">JPEG, PNG, WebP or PDF (first page used).</label>
          <input id="doc" type="file" accept="image/jpeg,image/png,image/webp,application/pdf" disabled={busy}
            onChange={(e) => onDocument(e.target.files?.[0])} />
        </div>
      )}

      {step === "video" && session && (
        <Recorder challenge={session.challenge} durationMs={session.duration_ms} onDone={onVideo} />
      )}

      {step === "submitting" && <p>Uploading and queueing analysis…</p>}
    </section>
  );
}
