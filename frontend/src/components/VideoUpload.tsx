import { useEffect, useState } from "react";

/** Choose a pre-recorded selfie video (MP4/WebM), preview it, then send it for analysis. */
export default function VideoUpload({ maxBytes, onDone }: { maxBytes: number; onDone: (f: File) => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  function pick(f: File | undefined) {
    setErr(null);
    if (!f) return;
    if (!/^video\/(mp4|webm|quicktime)$/.test(f.type) && !/\.(mp4|webm|mov)$/i.test(f.name))
      return setErr("Please choose an MP4, WebM or MOV video.");
    if (f.size > maxBytes) return setErr(`The video is larger than ${Math.round(maxBytes / 1048576)} MB.`);
    if (url) URL.revokeObjectURL(url);
    setFile(f);
    setUrl(URL.createObjectURL(f));
  }

  return (
    <div className="card space-y-4">
      <h1 className="text-xl font-semibold">Upload a recorded video</h1>
      {!file ? (
        <label htmlFor="vidfile"
          onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files?.[0]); }}
          className={`flex flex-col items-center justify-center gap-2 rounded-2xl border-2 border-dashed p-12 text-center cursor-pointer transition
            ${drag ? "border-blue-500 bg-blue-50" : "border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/50"}`}>
          <span className="text-4xl" aria-hidden>🎬</span>
          <span className="font-medium">Drop a video here or click to choose</span>
          <span className="text-sm text-slate-500">MP4, WebM or MOV · 5–30 s works best (only the first 30 s are analysed) · max {Math.round(maxBytes / 1048576)} MB</span>
        </label>
      ) : (
        <div className="space-y-3">
          <div className="rounded-2xl overflow-hidden bg-slate-900 aspect-video">
            <video src={url ?? undefined} controls playsInline className="w-full h-full object-contain" aria-label="Selected video" />
          </div>
          <p className="text-sm text-slate-600">{file.name} · {(file.size / 1048576).toFixed(1)} MB</p>
          <div className="flex flex-wrap gap-3">
            <button onClick={() => onDone(file)} className="btn-primary">✓ Analyse this video</button>
            <button onClick={() => { setFile(null); setUrl(null); }} className="btn-ghost">Choose another</button>
          </div>
        </div>
      )}
      <input id="vidfile" type="file" className="sr-only" accept="video/mp4,video/webm,video/quicktime,.mp4,.webm,.mov"
        onChange={(e) => pick(e.target.files?.[0])} />
      {err && <p role="alert" className="text-rose-700 text-sm">{err}</p>}
      <p className="text-xs text-slate-500">For the liveness check, the video should show you blinking and turning your head left, right,
        up and down. Uploaded videos are marked as <em>uploaded</em> (not recorded live) in the monitoring dashboard.</p>
    </div>
  );
}
