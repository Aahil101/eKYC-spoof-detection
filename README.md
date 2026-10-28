# codmav: Video-KYC fraud detection with drift and novel-attack monitoring

Research prototype. **All six planned milestones are complete.** Every score comes from models trained on public datasets on Kaggle, and every result in the docs comes from a reproducible run. Nothing is hardcoded. Negative results are reported as such.

- Results: `docs/EXPERIMENTS.md` (all tables, with reproduce commands) · paper draft: `docs/PAPER_OUTLINE.md`
- Design: `docs/ARCHITECTURE.md`, `docs/MODULES.md`, `docs/FUSION.md`, `docs/MONITORING.md` · citations: `docs/RESEARCH_NOTES.md`
- Training on Kaggle: `docs/KAGGLE.md` · Privacy: `docs/PRIVACY.md` · handoff for coding agents: `AGENTS.md`

## What it does

**Dashboard 1 (KYC).** Consent → ID upload → selfie video with random challenges → five modules → calibrated fusion → ACCEPT / HUMAN_REVIEW / REJECT, with per-module scores, confidence and the capture-quality vector.

| module | method | headline test result |
|---|---|---|
| M1 document forgery | ELA/DCT/noise forensics + calibrated head | AUC 1.00 on seen designs, **0.55 on unseen designs** |
| M2 deepfake | frozen CLIP ViT-B/16 + spectrum + temporal cues, linear probe | FF++ 0.871, Celeb-DF (cross-dataset) 0.769 |
| M3 replay/PAD | colour LBP/moiré + GBM (CelebA-Spoof + MSU-MFSD video) | MSU-MFSD 0.914, CelebA-Spoof 0.979 |
| M4 liveness | rPPG (POS/CHROM) + blink + challenge-response | MSU-MFSD 0.691 (weak) |
| M5 face match | ArcFace (InsightFace, non-commercial) + landmark ratios | Celeb-DF identity pairs 0.995 |
| Fusion | GBM stacking + quality (F3); thresholds and veto chosen from data | AUC 0.963 ± 0.005 (5 seeds) |

**Dashboard 2 (MLOps monitoring).** It runs per-feature and multivariate drift tests against a **subject-level empirical null** (iid tests raise 100% false alarms; this null raises 10%), plus conformal novelty scoring. A quality-conditional rule then labels each window NONE / QUALITY_DRIFT / CONCEPT_DRIFT / OUTPUT_DRIFT / NOVEL_ATTACK / MIXED and emits actionable recommendations.
- **Results:** 0% false novelty in all scenarios; an OOD-only monitor reaches 95% false novelty under domain drift. All 5 controlled corruptions are detected and attributed to the right quality feature.
- **Retraining:** acting on the recommendation recovers 17–29% of the lost M2 AUC (E8).
- **Limitation:** truly new deepfake methods are **not** reliably detected (E6/E6b).
## Quick start (laptop, CPU)

Python 3.11 or 3.12, Node 20+.

```
python -m venv .venv && .venv\Scripts\activate          # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements/dev.txt
copy .env.example .env                                   # defaults: DEMO_MODE=1, SQLite
uvicorn backend.app.main:get_app --factory --port 8000

cd frontend && npm install && npm run dev                # http://127.0.0.1:5173
```

ML stack and weights: see `AGENTS.md` "How to run" (torch CPU, `requirements/ml.txt`, `python -m ml.common.weights fetch`). Trained model bundles are not in git; rebuild `kyc-v0.4` from the Kaggle outputs (`AGENTS.md`). Without a bundle, the modules report `UNAVAILABLE`.

Tests: `python -m pytest` (92 tests). Training plumbing smoke run: `python -m training.run --config training/configs/selftest.yaml --mode smoke`.

## Docker

```
docker compose up --build                     # http://127.0.0.1:8080
docker compose --profile postgres up --build  # also starts Postgres; set DATABASE_URL in .env
```

Ports are bound to 127.0.0.1 only.

## Security

`DEMO_MODE=1` (the default) leaves the admin/monitor API **unauthenticated**. It is reachable from localhost only, the UI shows a banner, and the server must not be exposed to a network. With `DEMO_MODE=0` the server refuses to start unless `ADMIN_API_KEY` is set; admin calls then need `X-API-Key`. KYC calls are scoped by a per-session token.

## Repository layout

```
backend/      FastAPI app (config, crypto, media store, DB, pipeline, auth, API)
ml/           modules M1-M5, features, heads, fusion, attribution embedding, model registry
training/     job runner, checkpoints, configs, dataset registry, Kaggle notebook generator, weight import
monitoring/   drift, novelty, drift-vs-novelty disambiguation, recommendations
experiments/  paper tables and experiments E1-E10 (results in docs/EXPERIMENTS.md)
frontend/     React + Vite + Tailwind
config/       pipeline.yaml, policy.yaml (thresholds null until calibrated)
docs/         research notes, architecture, experiment plan, Kaggle, privacy
tests/        pytest suite (synthetic media only)
```

Figures: `docs/figures/e2_reliability.png` (calibration).
