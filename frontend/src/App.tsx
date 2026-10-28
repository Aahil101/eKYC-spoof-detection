import { useEffect, useState } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { getPublicConfig, type PublicConfig } from "./api";
import KycFlow from "./pages/KycFlow";
import SessionResult from "./pages/SessionResult";
import Monitor from "./pages/Monitor";

const tab = ({ isActive }: { isActive: boolean }) =>
  `px-4 py-2 rounded-xl text-sm font-medium transition ${isActive ? "bg-white text-slate-900 shadow" : "text-slate-200 hover:text-white hover:bg-white/10"}`;

export default function App() {
  const [cfg, setCfg] = useState<PublicConfig | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    getPublicConfig().then(setCfg).catch((e) => setErr(String(e)));
  }, []);

  return (
    <div className="min-h-screen">
      <header className="bg-gradient-to-r from-slate-900 via-indigo-950 to-slate-900 text-white">
        <div className="max-w-6xl mx-auto px-6 py-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-blue-500 to-violet-500 grid place-items-center text-lg font-bold" aria-hidden>
              ✓
            </div>
            <div>
              <p className="font-semibold text-lg leading-tight">codmav Video-KYC</p>
              <p className="text-xs text-slate-300">Deepfake · replay · liveness · face match · document checks</p>
            </div>
          </div>
          <nav aria-label="Main" className="flex gap-2 bg-white/10 rounded-2xl p-1">
            <NavLink to="/" end className={tab}>KYC verification</NavLink>
            <NavLink to="/monitor" className={tab}>Drift &amp; novel-attack monitor</NavLink>
          </nav>
        </div>
        {(cfg?.demo_mode || cfg?.placeholder_bundle) && (
          <div className="bg-amber-400/15 border-t border-amber-300/30 text-amber-100 text-xs">
            <div className="max-w-6xl mx-auto px-6 py-1.5 flex flex-wrap gap-x-6">
              {cfg?.demo_mode && <span role="alert"><strong>DEMO MODE</strong>: local testing only, admin API unauthenticated, do not expose to a network.</span>}
              {cfg?.placeholder_bundle && <span><strong>PLACEHOLDER MODELS</strong>: no trained models loaded, no scores are produced.</span>}
              {cfg && !cfg.placeholder_bundle && <span>Models: bundle {cfg.bundle_version}</span>}
            </div>
          </div>
        )}
      </header>
      <main className="max-w-6xl mx-auto p-6">
        {err && <p role="alert" className="card text-rose-700">Backend unreachable: {err}</p>}
        {cfg && (
          <Routes>
            <Route path="/" element={<KycFlow cfg={cfg} />} />
            <Route path="/session/:id" element={<SessionResult />} />
            <Route path="/monitor" element={<Monitor />} />
          </Routes>
        )}
      </main>
      <footer className="max-w-6xl mx-auto px-6 pb-8 text-xs text-slate-500">
        Research prototype. Scores come from models trained on public research datasets; they are not a guarantee.
      </footer>
    </div>
  );
}
