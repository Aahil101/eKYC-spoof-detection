// Thin typed client for the FastAPI backend.
export type ChallengeStep = { kind: string; prompt: string; start_ms: number; end_ms: number };

export type PublicConfig = {
  demo_mode: boolean;
  consent_version: string;
  media_ttl_days: number;
  bundle_version: string;
  placeholder_bundle: boolean;
  max_document_bytes: number;
  max_video_bytes: number;
};

export type CreatedSession = { session_id: string; token: string; challenge: ChallengeStep[]; duration_ms: number };

export type ModuleOut = {
  module: string;
  title: string;
  status: "ok" | "placeholder" | "failed" | "not_applicable";
  score: number | null;
  confidence: number | null;
  model_version: string;
  latency_ms: number;
  message: string | null;
};

export type SessionOut = {
  session_id: string;
  status: string;
  created_at: string;
  bundle_version: string | null;
  decision: string | null;
  decision_reason: string | null;
  fused_score: number | null;
  error: string | null;
  challenge: ChallengeStep[];
  video_meta: Record<string, unknown> | null;
  quality: Record<string, number | null> | null;
  modules: ModuleOut[];
  placeholder: boolean;
};

async function check(res: Response) {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(`${res.status}: ${detail}`);
  }
  return res;
}

export async function getPublicConfig(): Promise<PublicConfig> {
  return (await check(await fetch("/api/config/public"))).json();
}

export async function createSession(consentVersion: string): Promise<CreatedSession> {
  const res = await fetch("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      consent: true,
      consent_version: consentVersion,
      client_meta: { user_agent: navigator.userAgent.slice(0, 200) },
    }),
  });
  return (await check(res)).json();
}

async function upload(s: CreatedSession, kind: "document" | "video", blob: Blob, filename: string) {
  const fd = new FormData();
  fd.append("file", blob, filename);
  await check(await fetch(`/api/sessions/${s.session_id}/${kind}`, {
    method: "POST",
    headers: { "X-Session-Token": s.token },
    body: fd,
  }));
}

export const uploadDocument = (s: CreatedSession, f: File) => upload(s, "document", f, f.name);
export const uploadVideo = (s: CreatedSession, b: Blob) =>
  upload(s, "video", b, b.type.includes("mp4") ? "selfie.mp4" : "selfie.webm");

export async function submit(s: CreatedSession) {
  await check(await fetch(`/api/sessions/${s.session_id}/submit`, {
    method: "POST",
    headers: { "X-Session-Token": s.token },
  }));
}

export async function getSession(id: string, token: string): Promise<SessionOut> {
  return (await check(await fetch(`/api/sessions/${id}`, { headers: { "X-Session-Token": token } }))).json();
}

export async function deleteSession(id: string, token: string) {
  await check(await fetch(`/api/sessions/${id}`, { method: "DELETE", headers: { "X-Session-Token": token } }));
}
