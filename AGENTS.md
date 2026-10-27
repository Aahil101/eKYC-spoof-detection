# AGENTS.md: handoff guide for coding agents

Read this first. It says where the project stands and how to continue without redoing work.

## Project
A research-grade Video-KYC fraud detection system with two dashboards (KYC decision; MLOps drift and novel-attack monitoring), built toward a paper (IEEE Access / PR Letters first). The full brief is in the docs:
`docs/ARCHITECTURE.md` (design), `docs/EXPERIMENT_PLAN.md` (E1–E10), `docs/RESEARCH_NOTES.md` (verified citations only, **do not invent citations**), `docs/MODULES.md` (M1–M5 as built), `docs/KAGGLE.md` (training workflow), `docs/DATASET_AUDIT.md`, `docs/PRIVACY.md`.

## Ground rules (from the owner, binding)
- Work in milestones. After each one, give a short summary covering what works, what is a placeholder, and what you need.
- **Never fake results.** No hardcoded scores. Untrained parts must report `placeholder` / `not_applicable` / `UNAVAILABLE`.
- Never use real people's ID documents. Store no raw biometrics in logs.
- The owner's laptop is CPU-only with little disk. **All training runs on Kaggle** (account `aahil1011`), with datasets attached online. Do not download datasets locally.
- Ask before big decisions (dataset changes, framework swaps, GPU assumptions).
- Secrets: the Kaggle token lives in `~/.kaggle/access_token` on the owner's machine. Never commit tokens, `.env`, `kaggle.json`, DBs, or `models/store/`.

## Milestones
| # | Scope | State |
|---|---|---|
| 1 | Research + plan docs | done |
| 2 | Skeleton, encrypted session logging, Kaggle infra | done (Docker verified, Kaggle round trip verified) |
| 3 | Real modules M1–M5 + Kaggle feature extraction + head training | **in progress** (see below) |
| 4 | Fusion F0–F6, calibrated thresholds (E4), veto, Dashboard 1 polish | not started |
| 5 | Drift + novelty + drift-vs-novelty rule + Dashboard 2 | not started |
| 6 | Experiments E1–E10 + paper outline | not started |

### Milestone 3 status (as of commit time)
- Done: face pipeline (`ml/common/face.py`); extractors (`ml/features/`); heads and calibration (`ml/heads.py`); serving modules (`ml/modules.py`, wired into `backend/app/pipeline.py`); UI module views; Kaggle jobs (`training/feature_jobs.py`); configs `training/configs/extract_*.yaml`, `train_heads.yaml`; 62 tests passing.
- Kaggle extraction notebooks (`kaggle.com/code/aahil1011/codmav-feats-<name>`):
  - `sidtd` COMPLETE (2,222 items, 0 errors), `msu` COMPLETE (280 items, 2 undecodable), `celebdf` COMPLETE (1,830 items, 1 error)
  - `ffpp` COMPLETE (6,000 videos, 0 errors, 7.6 h). `celebaspoof` COMPLETE (v2 after fixing a lister that stalled >12 h on rglob over 1.1M files; 25,500 images, 30 min).
  - Heads M1/M2/M4/M5 trained (`codmav-train-heads-a` v3) and imported as bundle `kyc-v0.3` (local only, `models/bundles` is git-ignored; rebuild with the import_heads command below). Results are in docs/MODULES.md. **M1's AUC of 1.0 is a compression shortcut; fix it in E1.**
  - All 5 heads trained in `codmav-train-heads` v1 and imported as `kyc-v0.3` (0.3.0+84ab76b466).
  - **Open issues before milestone 4:** (1) M3 image?video gap (MSU cross AUC 0.43; false-rejects genuine video), so retrain with MSU train subjects. (2) M1 SIDTD compression shortcut. Both are described in docs/MODULES.md.
  - Heads must be trained with **scikit-learn 1.9.1** (pinned), because pickles must match the serving version.
  - **Re-package codmav-src before pushing any notebook that uses a new config.**
- **Next steps:**
  1. When all five `codmav-feats-*` are COMPLETE:
     ```
     python -m training.make_notebook --config training/configs/train_heads.yaml --owner aahil1011
     kaggle kernels push -p <ABS>/kaggle/build/train_heads
     ```
  2. Download `runs/train_heads/artifacts/*` and import every head into one bundle:
     ```
     kaggle kernels output aahil1011/codmav-train-heads -p <dl> --file-pattern "runs/.*/(artifacts/.*|run_manifest.json)"
     python -m training.import_heads --run-dir <dl>/runs/train_heads --bundle kyc-v0.3 --semver 0.3.0
     ```
  3. Set `MODEL_BUNDLE=kyc-v0.3`, run a live session, and record the metrics from `heads_report.json` in `docs/MODULES.md`.
- Local check: a laptop run of `train_heads` on the M4+M5 shards gave MSU-MFSD test AUC 0.69 (M4, weak, as expected) and Celeb-DF identity-verification test AUC 0.995 (M5). Treat these as preliminary until the Kaggle run.

## How to run
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements/dev.txt
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/ml.txt
python -m ml.common.weights fetch          # MediaPipe, ArcFace (non-commercial), Vosk -> models/store/
python -m pytest                           # 62 tests
uvicorn backend.app.main:get_app --factory --port 8000 ; cd frontend && npm install && npm run dev
docker compose up --build                  # http://127.0.0.1:8080 (image ~3 GB with the ML stack)
```

## Kaggle loop (details in docs/KAGGLE.md)
On Windows, pass **absolute** paths to `kaggle ... -p`.
```
python -m training.package_src --owner aahil1011
kaggle datasets version -p <ABS>/dist/codmav-src -m "<sha>"
python -m training.make_notebook --config training/configs/<job>.yaml --owner aahil1011
kaggle kernels push -p <ABS>/kaggle/build/<run_name>
```
Notebooks use code from the `aahil1011/codmav-src` dataset, so **re-package and version it after every code change** before pushing notebooks.

## Gotchas found so far
- OpenAI CLIP weights need the `ViT-B-16-quickgelu` config.
- Kaggle mounts datasets at `/kaggle/input/datasets/<owner>/<slug>`. Never glob `/kaggle/input` recursively, because it walks 100k+ files.
- Docker: MediaPipe needs `libegl1 libgles2` and the headless contrib OpenCV build.
- CelebA-Spoof `*_BB.txt` boxes are in a 224×224 reference frame (verified visually).
- Python: laptop/Docker 3.12, Kaggle 3.13.
