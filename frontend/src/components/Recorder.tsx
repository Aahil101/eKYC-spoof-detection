import { useEffect, useRef, useState } from "react";

function pickMime(): string {
  const options = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"];
  return options.find((m) => MediaRecorder.isTypeSupported(m)) ?? "";
}

const GUIDE = [
  { icon: "🙂", text: "Look straight at the camera for a few seconds" },
  { icon: "😉", text: "Blink naturally" },
  { icon: "⬅️", text: "Turn your head left" },
  { icon: "➡️", text: "Turn your head right" },
  { icon: "⬆️", text: "Look up" },
  { icon: "⬇️", text: "Look down" },
];

/** Free recording: the user starts and stops when they want (min/max bounds only). No timed prompts:
 *  the actions in the guide are detected from the video frames after upload. */
export default function Recorder({ maxMs, minMs = 5000, onDone }: {
  maxMs: number;
  minMs?: number;
  onDone: (b: Blob) => void;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recRef = useRef<MediaRecorder | null>(null);
  const timerRef = useRef<number | null>(null);
  const [ready, setReady] = useState(false);
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [clip, setClip] = useState<{ blob: Blob; url: string } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: { width: 1280, height: 720, facingMode: "user" }, audio: true })
      .then((s) => {
        streamRef.current = s;
        if (videoRef.current) videoRef.current.srcObject = s;
        setReady(true);
      })
      .catch((e) => setErr(`Camera/microphone not available: ${e.message}. You can still upload a video file below (demo mode).`));
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  function start() {
    const stream = streamRef.current;
    if (!stream) return;
    if (clip) URL.revokeObjectURL(clip.url);
    setClip(null);
    const mime = pickMime();
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 2_500_000 } : undefined);
    const chunks: Blob[] = [];
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    rec.onstop = () => {
      if (timerRef.current) clearInterval(timerRef.current);
      setRecording(false);
      const blob = new Blob(chunks, { type: rec.mimeType || "video/webm" });
      setClip({ blob, url: URL.createObjectURL(blob) });
    };
    const t0 = performance.now();
    timerRef.current = window.setInterval(() => {
      const t = performance.now() - t0;
      setElapsed(t);
      if (t >= maxMs && rec.state === "recording") rec.stop();
    }, 100);
    rec.start(250);
    recRef.current = rec;
    setElapsed(0);
    setRecording(true);
  }

  function stop() {
    if (recRef.current?.state === "recording") recRef.current.stop();
  }

  const secs = (elapsed / 1000).toFixed(1);
  const canStop = elapsed >= minMs;

  return (
    <div className="grid lg:grid-cols-[1fr_300px] gap-6">
      <div className="card space-y-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold">Record your selfie video</h1>
          {recording && (
            <span className="flex items-center gap-2 text-rose-600 font-semibold tabular-nums">
              <span className="rec-dot h-3 w-3 rounded-full bg-rose-600" aria-hidden /> REC {secs} s
            </span>
          )}
        </div>
        {err && <p role="alert" className="text-rose-700 text-sm">{err}</p>}
        <div className="relative rounded-2xl overflow-hidden bg-slate-900 aspect-video">
          {clip ? (
            <video src={clip.url} controls playsInline aria-label="Your recording" className="w-full h-full object-contain" />
          ) : (
            <video ref={videoRef} autoPlay muted playsInline aria-label="Camera preview"
              className="w-full h-full object-cover -scale-x-100" />
          )}
          {!clip && (
            <div className="pointer-events-none absolute inset-0 grid place-items-center" aria-hidden>
              <div className={`h-[70%] aspect-[3/4] rounded-[50%] border-4 ${recording ? "border-rose-400/80" : "border-white/50"} border-dashed`} />
            </div>
          )}
        </div>
        {recording && (
          <div className="h-2 rounded-full bg-slate-200 overflow-hidden" role="progressbar" aria-valuemin={0}
            aria-valuemax={maxMs / 1000} aria-valuenow={Math.round(elapsed / 1000)} aria-label="Recording length">
            <div className="h-full bg-rose-500 transition-all" style={{ width: `${Math.min(100, (elapsed / maxMs) * 100)}%` }} />
          </div>
        )}
        <div className="flex flex-wrap items-center gap-3">
          {!recording && !clip && (
            <button disabled={!ready} onClick={start} className="btn-danger">● Start recording</button>
          )}
          {recording && (
            <button disabled={!canStop} onClick={stop} className="btn-primary">
              ■ Stop {canStop ? "" : `(at least ${Math.ceil((minMs - elapsed) / 1000)} s more)`}
            </button>
          )}
          {clip && (
            <>
              <button onClick={() => onDone(clip.blob)} className="btn-primary">✓ Use this video & analyse</button>
              <button onClick={start} className="btn-ghost">↺ Record again</button>
              <span className="text-sm text-slate-500">{secs} s recorded</span>
            </>
          )}
          <span className="text-xs text-slate-500 ml-auto">Record at your own pace · {minMs / 1000}–{maxMs / 1000} s</span>
        </div>
      </div>

      <aside className="card space-y-3">
        <h2 className="font-semibold">During the recording</h2>
        <p className="text-sm text-slate-600">No timed prompts. Do these in any order, at your own pace. The system finds them in the video frames afterwards.</p>
        <ol className="space-y-2">
          {GUIDE.map((g, i) => (
            <li key={g.text} className="flex items-center gap-3 rounded-xl bg-slate-50 ring-1 ring-slate-200 px-3 py-2 text-sm">
              <span className="text-xl" aria-hidden>{g.icon}</span>
              <span><span className="text-slate-400 mr-1">{i + 1}.</span>{g.text}</span>
            </li>
          ))}
        </ol>
        <p className="text-xs text-slate-500">Good light, face inside the oval, no sunglasses or mask.</p>
      </aside>
    </div>
  );
}
