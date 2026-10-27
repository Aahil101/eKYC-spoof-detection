import { useEffect, useState } from "react";
import { Link, Route, Routes } from "react-router-dom";
import { getPublicConfig, type PublicConfig } from "./api";
import KycFlow from "./pages/KycFlow";
import SessionResult from "./pages/SessionResult";
import Monitor from "./pages/Monitor";

export default function App() {
  const [cfg, setCfg] = useState<PublicConfig | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getPublicConfig().then(setCfg).catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="min-h-screen">
      {cfg?.demo_mode && (
        <div role="alert" className="bg-amber-100 border-b border-amber-300 px-4 py-2 text-sm text-amber-900">
          <strong>DEMO MODE</strong>: unauthenticated admin API, localhost only. Do not expose to a network.
        </div>
      )}
      {cfg?.placeholder_bundle && (
        <div role="status" className="bg-slate-200 border-b border-slate-300 px-4 py-2 text-sm">
          <strong>PLACEHOLDER MODELS</strong>: no trained models are loaded (bundle {cfg.bundle_version}). The
          system records and logs sessions but produces <em>no scores and no decisions</em>.
        </div>
      )}
      <header className="bg-white border-b px-6 py-4 flex items-center justify-between">
        <Link to="/" className="font-semibold text-lg">codmav Video-KYC</Link>
        <nav aria-label="Main" className="text-sm text-slate-600">
          Research prototype · Monitoring dashboard: milestone 5
        </nav>
      </header>
      <main className="max-w-6xl mx-auto p-6">
        {err && <p role="alert" className="text-red-700">Backend unreachable: {err}</p>}
        {cfg && (
          <Routes>
            <Route path="/" element={<KycFlow cfg={cfg} />} />
            <Route path="/session/:id" element={<SessionResult />} />
            <Route path="/monitor" element={<Monitor />} />
          </Routes>
        )}
      </main>
    </div>
  );
}
