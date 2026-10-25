import { useEffect, useRef, useState } from "react";
import type { ChallengeStep } from "../api";

function pickMime(): string {
  const options = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
  return options.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
}

/** Records audio+video for durationMs while showing the server-issued challenge prompts on time. */
export default function Recorder({ challenge, durationMs, onDone }: {
  challenge: ChallengeStep[];
  durationMs: number;
  onDone: (b: Blob) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" }, audio: true })
      .then((s) => {
        streamRef.current = s;
        if (videoRef.current) videoRef.current.srcObject = s;
        setReady(true);
      })
      .catch((e) => setErr(`Camera/microphone not available: ${e.message}`));
    return () => streamRef.current?.getTracks().forEach((t) => t.stop());
  }, []);

  function record() {
    const stream = streamRef.current;
    if (!stream) return;
    const mime = pickMime();
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 2_500_000 } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      setRecording(false);
      onDone(new Blob(chunks, { type: rec.mimeType || "video/webm" }));
    };
    const t0 = performance.now();
    const timer = setInterval(() => setElapsed(performance.now() - t0), 100);
    rec.start(250);
    setRecording(true);
    setTimeout(() => {
      clearInterval(timer);
      rec.stop();
    }, durationMs);
  }

  const current = challenge.find((c) => elapsed >= c.start_ms && elapsed < c.end_ms);

  return (
    <div className="bg-white rounded-lg shadow p-6 space-y-4">
      <h1 className="text-xl font-semibold">Record selfie video</h1>
      <p>Face the camera in good light. When recording starts, follow each prompt as it appears.</p>
      {err && <p role="alert" className="text-red-700">{err}</p>}
      <div className="relative">
        <video ref={videoRef} autoPlay muted playsInline aria-label="Camera preview"
          className="w-full rounded bg-black -scale-x-100" />
        {recording && (
          <div aria-live="assertive" className="absolute inset-x-0 top-4 text-center">
            <span className="inline-block bg-black/70 text-white text-2xl font-semibold px-4 py-2 rounded">
              {current?.prompt ?? "Look at the camera"}
            </span>
          </div>
        )}
      </div>
      {recording && (
        <progress className="w-full" max={durationMs} value={elapsed} aria-label="Recording progress" />
      )}
      <button disabled={!ready || recording} onClick={record}
        className="px-4 py-2 rounded bg-red-700 text-white disabled:opacity-50">
        {recording ? "Recording…" : `Start recording (${Math.round(durationMs / 1000)} s)`}
      </button>
    </div>
  );
}
