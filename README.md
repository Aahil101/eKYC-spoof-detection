# codmav: Video-KYC fraud detection with drift and novel-attack monitoring

Research prototype. **Status: milestone 2 of 6** (skeleton, data logging, Kaggle training infrastructure). **No trained models yet:** every module is a labelled placeholder, and the system returns `decision = UNAVAILABLE` with no scores. Nothing is hardcoded or faked.

- Plan and design: `docs/ARCHITECTURE.md`, `docs/EXPERIMENT_PLAN.md`, `docs/RESEARCH_NOTES.md`
- Training on Kaggle: `docs/KAGGLE.md` · Privacy: `docs/PRIVACY.md`

## What works now

- KYC flow: consent → ID upload (JPEG/PNG/WebP/PDF) → in-browser selfie recording with server-issued random challenge prompts → asynchronous processing → result page with per-module status, the capture-quality vector, and "delete my data".
- Backend: FastAPI, encrypted media store (AES-256-GCM, key rotation), SQLite by default or Postgres via a profile, session/module/quality/embedding/label/drift/alert tables, retention purge, versioned model bundles.
- Quality vector: resolution, FPS and jitter, bitrate, bits per pixel, Laplacian variance, Tenengrad, brightness, contrast, clipping, noise σ, and ID JPEG blockiness. Face size and head pose are NaN until milestone 3.
- Training infrastructure: config-driven jobs (smoke on CPU / full on Kaggle), crash-safe checkpoints with automatic resume across Kaggle sessions, a notebook generator, a source packager for offline notebooks, and a weight importer that writes into versioned bundles.

## Quick start (laptop, CPU)

Python 3.11 or 3.12, Node 20+.

```
python -m venv .venv && .venv\Scripts\activate          # Windows (use source .venv/bin/activate on Linux/macOS)
pip install -r requirements/dev.txt
copy .env.example .env                                   # defaults: DEMO_MODE=1, SQLite
uvicorn backend.app.main:get_app --factory --port 8000

cd frontend && npm install && npm run dev                # http://127.0.0.1:5173
```

Tests: `python -m pytest` (43 tests). Training plumbing smoke run: `python -m training.run --config training/configs/selftest.yaml --mode smoke`.

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
ml/           module interface, media decoding, quality vector, placeholders, model registry
training/     job runner, checkpoints, configs, dataset registry, Kaggle notebook generator, weight import
monitoring/   drift + novelty (milestone 5)
experiments/  reproducible experiments (milestone 6)
frontend/     React + Vite + Tailwind
config/       pipeline.yaml, policy.yaml (thresholds null until calibrated)
docs/         research notes, architecture, experiment plan, Kaggle, privacy
tests/        pytest suite (synthetic media only)
```

Screenshots will be added once milestone 4 produces a real result page.
