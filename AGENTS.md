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
| 3 | Real modules M1–M5 + Kaggle feature extraction + head training | **done** (heads v3; docs/MODULES.md) |
| 4 | Fusion F0–F6, data-driven thresholds/veto, Dashboard 1 | **done** (docs/FUSION.md) |
| 5 | Drift + novelty + drift-vs-novelty rule + Dashboard 2 | **done**; drift works, novel-attack detection does NOT (docs/MONITORING.md) |
| 6 | Experiments E1–E10 + paper outline | **done** (docs/EXPERIMENTS.md, docs/PAPER_OUTLINE.md) |

All six planned milestones are complete. Remaining work is research follow-up (see "Open items").

### Served bundle `kyc-v0.4` (models/bundles is git-ignored: rebuild from Kaggle outputs)
```
kaggle kernels output aahil1011/codmav-train-heads     -p <dl1> --file-pattern "runs/train_heads/(artifacts/.*|run_manifest.json)"
kaggle kernels output aahil1011/codmav-train-fusion    -p <dl2> --file-pattern "runs/train_fusion/(artifacts/.*|run_manifest.json)"
kaggle kernels output aahil1011/codmav-build-reference -p <dl3> --file-pattern "runs/build_reference/(artifacts/.*|run_manifest.json)"
python -m training.import_heads   --run-dir <dl1>/runs/train_heads --bundle kyc-v0.4 --semver 0.4.0
python -m training.import_weights --run-dir <dl2>/runs/train_fusion --module FUSION --bundle kyc-v0.4 --semver 0.4.0 --artifacts fusion.pkl
python -m training.import_weights --run-dir <dl3>/runs/build_reference --module MONITOR --bundle kyc-v0.4 --semver 0.4.0 --artifacts reference_served.pkl --allow-incomplete
```
Then set `MODEL_BUNDLE=kyc-v0.4`. The heads come from `codmav-train-heads` v3 (fv2: M3 with video data, M1 normalised documents), fusion from `codmav-train-fusion` v4 (F3 stacking, out-of-fold thresholds, rule-of-three veto), and the monitor reference from `codmav-build-reference` v2 (subject-level empirical null, final drift-vs-novelty rule).

### Pipelines (in order; all on Kaggle, CPU unless noted)
1. `codmav-fetch-sidtd-templates` → `codmav-feats-{sidtd,msu,celebaspoof}` + `codmav-feats-{ffpp,celebdf}` (GPU)
2. `codmav-train-heads` → `codmav-train-fusion` → `codmav-build-reference`
3. `codmav-feats-ffpp-corrupt` (GPU, E5)
4. Local analysis on downloaded feature shards (≈200 MB, temp only): `experiments/paper_tables.py` (E1/E3), `e2_e10.py`, `e5_corruptions.py`, `e6_attribution.py`, `e8_closed_loop.py`, `e9_latency.py`. The commands are in `docs/EXPERIMENTS.md`.

### Key findings to keep in mind
- **M1** fails on unseen document designs (leave-one-type-out AUC 0.55). The SIDTD AUC of 1.0 is only for seen designs.
- **M3** needs video training data: image-only AUC on video was 0.43; with video data, 0.914.
- **M2** loses 0.13–0.24 AUC under every capture corruption (E5a).
- **Fusion:** F3 > F1 is significant; quality gating (F4) is not better on clean data.
- **Monitoring** needs a subject-level empirical null; iid tests give 100% false alarms.
- **Novelty** of a held-out FF++ method is barely detectable. The supervised-contrastive attribution embedding (`ml/attribution.py`, novelty space `supcon`) raises sample AUROC 0.58 → 0.64 and removes most of the domain confound, but window-level detection is still 0% (E6b).
- **Closed loop:** augmenting M2 with the recommended corruption recovers 17–29% of the lost AUC (E8a). Under corruption, F1 is more robust than F3/F4 (E8b).
- **E10:** no demographic labels exist. Capture subgroups show large gaps (laptop genuine false-reject 0.45 vs Android 0.15).
- `build_reference` params: `attribution_embedding` (default true) adds the `supcon64` candidate to the leave-one-class-out selection; `novelty_space: supcon64/knn` forces a space (ablation).

### Open items (priority, research follow-up)
1. Novelty: fine-tune a forgery backbone end-to-end with SupCon (E6b uses frozen 541-d features), and evaluate on unseen generators (DF40 slice: SadTalker/SimSwap; DFDC).
2. Demographic audit with a labelled dataset (licence permitting).
3. A real consented KYC pilot, to remove the independence assumption of composite sessions.
4. Downscale/JPEG windows are still labelled MIXED in 30–40% of cases (E5b). Improve the quality-reweighting model.
5. Possible plug-ins: TruFor for M1, SBI-EfficientNet for M2 (config `m2_sbi_effnet.yaml`, not implemented).
6. Latency on GPU / Docker (E9 is laptop CPU only).
### KYC UX (free recording, after milestone 6)
- `config/pipeline.yaml` `challenge.mode: free`: no timed prompts. The user records 5–30 s at their own pace.
- `ml/features/liveness.detect_actions` finds blink / turn left / turn right / look up / look down in 8 fps frames over the whole video, using geometric pose relative to the neutral pose. The thresholds are heuristic and **unvalidated**, so missing actions cap ACCEPT at HUMAN_REVIEW.
- Model frames: the 16 most frontal of 32 candidates spread over the whole video (`frontal_pool`), so head turns don't push the models off their mostly frontal training distribution. The defaults in `VIDEO_DEFAULTS` keep Kaggle extraction identical to training.
- The result page shows the video verdict (REAL PERSON / DEEPFAKE DETECTED / SPOOF SUSPECTED / UNCERTAIN), plus per-frame deepfake scores from the M2 head applied to single-frame CLIP embeddings. These are indicative only, because the head was trained on 16-frame means.
- In demo mode the ID document can be skipped. M1/M5 are then not applicable and the decision is UNAVAILABLE: the fusion was fitted only on sessions that include a document, and it outputs ~1e-6 without one.
- Video upload: a Record / Upload tab, enabled in demo mode or with `ALLOW_VIDEO_UPLOAD=1`. Uploading bypasses live capture (video-injection risk). `?capture=upload|camera` is stored in `client_meta` and shown on Dashboard 2.
- Dashboard 2 live feed: `GET /api/monitor/sessions?source=live|simulated|all` (`monitoring/service.live_sessions`), auto-refreshed every 4 s. Per session it shows module scores, the verdict and the actions. It also shows a capture check: quality features outside the central 98% of the reference. And it shows the conformal artifact-novelty p-value. Window-level drift still needs ≥ max(30, W/4) sessions, because the null is calibrated for windows of W.

### History notes
- Heads must be trained with **scikit-learn 1.9.1** (pinned), because pickles must match the serving version.
- **Re-package `codmav-src` and version the dataset before pushing any notebook that uses new code or configs.**
- Local check: a laptop run of `train_heads` on the M4+M5 shards gave MSU-MFSD test AUC 0.69 (M4, weak, as expected) and Celeb-DF identity-verification test AUC 0.995 (M5). Treat these as preliminary until the Kaggle run.

## How to run
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements/dev.txt
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/ml.txt
python -m ml.common.weights fetch          # MediaPipe, ArcFace (non-commercial), Vosk -> models/store/
python -m pytest                           # 92 tests
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
