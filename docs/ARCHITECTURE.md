# Architecture

Status: **milestone 2 implemented** (skeleton, logging, training infrastructure). Modules, fusion and monitoring are still the plan. Section 17 lists decisions and deviations. Bracketed numbers like [21] refer to entries in `RESEARCH_NOTES.md`.

## 1. Design principles

1. **CPU-first.** The host has no NVIDIA GPU. Every module has a CPU path made of frozen pretrained backbones, hand-crafted forensic features and light heads (logistic regression or gradient boosting). The heads train in minutes on CPU. This is not only a compromise: UnivFD [21] shows that frozen features with a linear probe generalize across generators better than full fine-tuning. Heavy backbones (TruFor [4], CDCN [32], SBI-EfficientNet [20,24]) are optional plug-ins behind the same interface.
2. **One interface for every module.** `predict(session) -> ModuleOutput(score, confidence, features, embedding, explanations, model_version)`. Fusion, logging and monitoring never see module internals.
3. **No fake numbers.** Any component without trained weights reports `status="placeholder"`. The API returns that status, the UI shows a grey "PLACEHOLDER" badge, and fusion excludes the module, with the exclusion logged. Scores are never hardcoded.
4. **Everything versioned.** Every score is stamped with `model_bundle_version` (hash of module weights, calibrators, fusion and threshold config). Drift alerts reference that version.
5. **Privacy by default.** Raw media is encrypted at rest with a retention TTL. Logs hold features and artifact embeddings. Identity templates (face-match embeddings) are stored separately, encrypted, and purged with the session.

## 2. System overview

```
Browser (React)                      Backend (FastAPI)                         Storage
────────────────                     ─────────────────                         ───────
Consent ─► ID upload ─► Selfie rec.  POST /sessions            ─► MediaStore (AES-GCM files, TTL)
 (MediaRecorder, random challenges)  POST /sessions/{id}/run   ─► JobRunner (process pool)
                                          │                                        
                                          ▼                                        
                                     Preprocess: decode (PyAV), frame sample,     
                                     face detect + 478 landmarks (MediaPipe),     
                                     doc rectify, Quality vector q                
                                          │                                        
                     ┌──────────┬─────────┼──────────┬──────────┐                  
                     M1 doc     M2 deepfake M3 replay M4 liveness M5 biometrics    
                     └──────────┴─────────┼──────────┴──────────┘                  
                                          ▼                                        
                                     Calibrators (TS / isotonic)                   
                                          ▼                                        
                                     Fusion (F1..F6, configured)                   
                                          ▼                                        
                                     Decision policy (+ hard veto) ─► SessionLog (SQL)
Dashboard 1 ◄── GET /sessions/{id}                                                 
                                                                                    
Dashboard 2 ◄── GET /monitor/*   ◄── MonitorWorker (scheduled) ◄── reads SessionLog
                                     drift tests, OOD scores, HDBSCAN,            
                                     drift-vs-novelty classifier, recommender      
                                     ─► drift_reports, alerts (SQL)                
```

## 3. Repository layout

```
codmav/
├─ backend/            FastAPI app: api/, schemas/, services/ (jobs, media, crypto), db/ (SQLAlchemy models, alembic)
├─ ml/
│  ├─ common/          interface.py (Module protocol, ModuleOutput), preprocessing (video, face, doc), quality.py
│  ├─ m1_document/     ela.py, dct.py, noise.py, recapture_fft.py, mrz.py, model.py
│  ├─ m2_deepfake/     clip_probe.py, frequency.py, temporal.py, sbi.py (pseudo-fake generator), model.py
│  ├─ m3_replay/       ida.py, color_lbp.py, moire.py, motion_corr.py, model.py
│  ├─ m4_liveness/     rppg_pos.py, rppg_chrom.py, blink_ear.py, headpose.py, challenges.py, model.py
│  ├─ m5_biometrics/   geometry.py, stability.py, facematch.py, model.py
│  ├─ calibration/     temperature.py, isotonic.py, ece.py
│  ├─ fusion/          weighted.py, stacking.py, gated.py, evidential.py, policy.py, thresholds.py
│  ├─ explain/         gradcam.py, attributions.py
│  └─ registry/        bundle.py (versioned model bundles + manifest), store/ (weights; git-ignored)
├─ monitoring/
│  ├─ drift/           univariate.py (KS, Wasserstein, PSI, JS), mmd.py, c2st.py, streaming.py (ADWIN, PH, DDM, EDDM), fdr.py
│  ├─ novelty/         mahalanobis.py, knn.py, energy.py, odin.py, svdd.py, iforest_lof.py, openmax.py, conformal.py, clustering.py
│  ├─ disambiguation/  dn_classifier.py  (drift vs novelty rule — core contribution)
│  ├─ recommend/       sensitivity.py, recommender.py
│  └─ worker.py        scheduled window evaluation
├─ frontend/           React + Vite + Tailwind + Recharts (kyc/ and monitor/ routes)
├─ experiments/        configs/*.yaml, run.py, stream_simulator.py, corruptions.py, results/ (git-ignored)
├─ data/               README with download instructions; raw/ processed/ (git-ignored); synthetic/ generators
├─ tests/              unit + API tests (pytest)
├─ docs/               RESEARCH_NOTES, ARCHITECTURE, EXPERIMENT_PLAN, PAPER_OUTLINE (M6), PRIVACY
├─ docker-compose.yml  services: api, worker, frontend, (postgres profile)
└─ README.md
```

## 4. Common module interface

```python
class ModuleOutput(BaseModel):
    module: Literal["M1","M2","M3","M4","M5"]
    score: float               # calibrated P(genuine) in [0,1]; 1 = genuine
    raw_score: float           # pre-calibration score (logit or margin)
    confidence: float          # in [0,1]; 1 - uncertainty mass (see §6.3)
    features: dict[str, float] # named, logged, used by drift monitoring
    embedding: list[float] | None   # artifact-space embedding for novelty (not identity)
    explanations: dict         # heatmap refs, top feature contributions, failure reason code
    status: Literal["ok","placeholder","failed","not_applicable"]
    model_version: str
    latency_ms: float

class KYCModule(Protocol):
    name: str
    def predict(self, s: PreparedSession) -> ModuleOutput: ...
    def fit(self, ds: ModuleDataset, cfg: dict) -> None: ...   # head training only on CPU path
```

`PreparedSession` holds sampled frames (default 16 over the clip, plus a 30 fps strip for rPPG), face crops, the 478 landmarks per frame, the rectified ID image, the challenge script with timestamps, and the quality vector `q`.

## 5. Modules (CPU path; optional deep path in brackets)

| Module | Inputs | Features (logged) | Learner | Embedding for novelty | Explanations |
|---|---|---|---|---|---|
| **M1 Document** | rectified ID image | multi-quality ELA statistics [1]; DCT-histogram periodicity / blockiness [2,5]; noise-residual inconsistency across blocks [3]; FFT radial-spectrum peaks for moiré/recapture [8,33]; glare/specular ratio; MRZ check digits and MRZ↔printed field consistency [12] (OCR via Tesseract, optional); portrait-region presence | Gradient boosting (sklearn `HistGradientBoosting`) + isotonic calibration [55] [optional: TruFor [4] / CAT-Net [2] plug-in] | standardized forensic feature vector | ELA map, block-noise map, Grad-CAM from a ResNet-18 linear-probe variant [85] |
| **M2 Deepfake** | 16 face crops (aligned via landmarks) | frozen CLIP ViT-B/16 image embedding → linear probe [21]; azimuthal spectrum bins [23]; temporal jitter of embeddings and landmarks [22]; blink statistics [26]; rPPG coherence across face regions (shared with M4) [25] | Logistic regression on [CLIP ⊕ handcrafted] + temperature scaling [54]; trained on dataset fakes + SBI pseudo-fakes [20] [optional: SBI EfficientNet-B4 [20,24] on GPU] | mean-pooled L2-normalized CLIP embedding (512-d) | Grad-CAM on the last ViT block (token reshape) [85]; top contributing feature groups |
| **M3 Replay** | full frames + face crops | IDA vector: specular, blur, chromatic moments, colour diversity [34]; colour LBP in HSV/YCbCr [35]; FFT moiré peaks [33]; face-vs-background optical-flow correlation [36]; screen-bezel/rectangle detection; gamma/saturation statistics | Gradient boosting + isotonic [optional: CDCN [32]] | standardized texture vector | moiré spectrum plot, flow-correlation plot |
| **M4 Liveness** | 30 fps face ROI strip, landmarks, challenge script | rPPG via POS [39] and CHROM [38] on forehead/cheek ROIs: HR-band SNR, peak prominence, plausible-HR flag, inter-ROI coherence [41]; EAR blink count and durations [42]; head-pose yaw/pitch trajectory vs the requested challenge; mouth-aperture activity during the "read digits" window; micro-motion energy | Logistic regression + temperature scaling; challenge compliance as a hard sub-score | none (liveness features feed drift only) | rPPG waveform and spectrum plot; challenge timeline with pass/fail |
| **M5 Biometrics** | ID portrait + frontal frames + landmarks | scale-invariant ratios: IPD/face-width, iris-diameter/IPD [45], jaw-width/face-length, eye-nose-mouth vertical ratios, face aspect ratio; per-ratio coefficient of variation across pose-filtered frames; ID-vs-selfie ratio deltas (frontal frames only); ArcFace cosine similarity [46] | Logistic regression on [cosine, ratio deltas, stability] + temperature scaling | none (identity data is never used for novelty) | landmark overlay, ratio table, similarity value |

**Quality vector q** (computed once per session, logged, used by gating and drift): video resolution, face bbox height (px and fraction), Laplacian variance and Tenengrad [47], mean brightness, RMS contrast, % clipped pixels, noise σ (MAD of a high-pass residual), JPEG blockiness on the ID, container bitrate / codec / GOP (PyAV), measured FPS and frame-time jitter, head-pose yaw/pitch/roll mean and std, BRISQUE (optional) [48], ID image resolution and skew.

## 6. Calibration, fusion and decision

### 6.1 Calibration
For each module, a calibration split is held out from training. Temperature scaling is used for logit heads [54] and isotonic regression for tree heads [55]. ECE (15 bins) and reliability diagrams are reported [56]. Calibrators are part of the versioned bundle.

### 6.2 Fusion strategies (all implemented; selected by config)
- **F0** mean of calibrated scores.
- **F1** fixed weighted sum of logits, with weights ∝ validation AUC−0.5 (sum-rule family [49]).
- **F2** logistic-regression stacking on `[logit(s_i), c_i]` [52].
- **F3** gradient-boosting stacking on `[logit(s_i), c_i, q]`.
- **F4** quality-gated mixture: `w = softmax(g_θ(q, c))`, `z = Σ w_i·logit(s_i) + b`. `g_θ` is a 2-layer MLP (16 hidden units) trained with BCE [50,53]. An optional monotonic rule prior caps a module's weight when its known-sensitive quality features fall outside the training range.
- **F5** evidential Dempster–Shafer: each module's (s_i, c_i) is mapped to a binomial opinion `(b_genuine, b_fraud, u)` with `u = 1 − c_i`, and opinions are combined with the reduced DS rule of TMC [57,58]. The output is a fused belief plus a fused uncertainty.
- **F6** F4 + F5: quality-gated discounting of opinions before DS combination.

The weights actually used per session (fixed weights, gate outputs, or DS discount factors) are logged and shown in Dashboard 1.

### 6.3 Module confidence
`confidence = 1 − u`. `u` is computed in one of two ways, chosen per module and documented: (a) an evidential head [57] where the model is an NN head, or (b) a quality-conditioned reliability estimate, i.e. the local validation accuracy of that module in the k-NN neighbourhood of `q`. Option (b) is CPU-cheap and model-agnostic. Both are evaluated, and confidence must be **predictive of correctness**, measured by AUROC of confidence vs correct/incorrect.

### 6.4 Decision policy
```
if any critical module (M2, M3) has P(fraud) ≥ τ_veto and confidence ≥ c_veto → REJECT (veto, reason = that module)
elif fused ≥ T_high → ACCEPT
elif fused ≥ T_low  → HUMAN_REVIEW
else                → REJECT (reason = module with largest negative contribution; user-facing message from a reason-code table)
```
`T_high` is chosen on the validation set so that attack acceptance (APCER over all attack types, and the worst per-type value) is ≤ the target, e.g. 1%. `T_low` is chosen so that bona fide rejection (BPCER) is ≤ the target, e.g. 2%. `τ_veto` is chosen so that genuine sessions are vetoed ≤ 0.1% of the time. Bootstrap 95% CIs are stored with each threshold. Every value lives in `config/policy.yaml` and in the bundle manifest, so changing it creates a new bundle version.

## 7. Session log schema (SQL; SQLite default, Postgres via compose profile)

- `sessions(id uuid, created_at, bundle_version, challenge_script json, decision, fused_score, fused_uncertainty, fusion_strategy, fusion_weights json, veto_module, client_meta json)`
- `module_outputs(session_id, module, score, raw_score, confidence, status, latency_ms, features json, explanations json)`
- `quality(session_id, …one column per q feature…)`
- `embeddings(session_id, space, vector blob)`: artifact spaces only (`m2_clip`, `m1_forensic`, `m3_texture`).
- `identity_templates(session_id, vector_encrypted blob, expires_at)`: M5 face embedding, encrypted, purged with media.
- `labels(session_id, label, attack_type, source='human_review'|'dataset', labelled_at)`: delayed labels.
- `drift_reports(id, window_start, window_end, ref_id, bundle_version, per_feature json, multivariate json, streaming json)`
- `alerts(id, created_at, type ∈ {quality_drift, concept_drift, output_drift, performance_drift, novel_attack, mixed}, severity, confidence, evidence json, recommendation text, acknowledged_by, status)`
- `reference_windows(id, bundle_version, created_at, session_ids or stored summary statistics)`

## 8. Monitoring (Dashboard 2 backend)

1. **Reference window.** These are the calibration/validation sessions for the active bundle. They are frozen with the bundle and summarized (bins, quantiles, class-conditional means and covariances).
2. **Current window.** Sliding, with configurable `size` (sessions or minutes) and `step`.
3. **Per-feature tests** on q, module features and module scores: KS, Wasserstein-1, PSI (quantile bins from the reference; critical values from the PSI sampling distribution [66]), Jensen–Shannon. p-values pass through BH-FDR [65]. Effect sizes are reported alongside p-values.
4. **Multivariate tests.** MMD with RBF kernel, median heuristic and permutation p-value [60], applied to (i) the q block, (ii) each artifact embedding after PCA to 32-d [62], and (iii) the score vector. C2ST [61] uses gradient boosting with a held-out-accuracy binomial test, and its feature importances provide attribution.
5. **Output drift.** Score histograms and the decision-rate shift (χ² test on accept/review/reject counts), plus BBSE-style confusion-based checks of prior shift [63].
6. **Streaming detectors** on the fused-score stream and the review-error stream: ADWIN [67], Page–Hinkley [70], DDM [68], EDDM [69]. Implemented in-house and cross-checked against `river` [71] in tests.
7. **Novelty scoring** per session in each artifact space: class-conditional Mahalanobis with shared covariance [72] (primary), kNN [75], energy [73], ODIN [74] (M2 head only), Deep SVDD [76], Isolation Forest [77], LOF [78], OpenMax-lite [79]. Each score is converted to a **conformal p-value** against a held-out reference split [81].
8. **Clustering.** HDBSCAN [82,83] runs on embeddings of current-window sessions with conformal p < α. Cluster stability, size, silhouette, distance to the nearest known class centroid, and the M2 mean uncertainty are recorded.
9. **Drift-vs-novelty classifier** (§9).
10. **Recommender** (§10).
11. **Alerts** get severity from effect size × persistence (number of consecutive windows) and confidence from test power. Each alert carries its evidence payload: histograms, statistic time series, cluster thumbnails (only if media retention allows, otherwise landmark/feature summaries) and nearest-class distance.

## 9. Drift vs novelty disambiguation (proposed method)

Features are split into blocks: **Q** (quality/sensor) and **A** (artifact embeddings and forensic features). Each window is tested for three hypotheses:

- **H_Q:** P_W(Q) = P_R(Q). Tested with MMD on Q plus per-feature BH.
- **H_A|Q:** P_W(A | Q) = P_R(A | Q). Reference samples are reweighted by a quality-only density ratio `w(q) = P_W(q)/P_R(q)`, estimated discriminatively with a classifier [64] (the C2ST classifier on Q only). A **weighted MMD** is then computed between the reweighted reference and the current window in A. If the A-shift is fully explained by the quality shift, H_A|Q is not rejected.
- **H_novel:** the fraction of current sessions with conformal novelty p < α equals α. Under H_novel this count is about Binomial(n, α) [81], so a one-sided binomial test applies.

Rule (outputs a label plus a confidence):

| H_Q rejected | H_A\|Q rejected | H_novel rejected | Coherent HDBSCAN cluster among outliers* | Label |
|---|---|---|---|---|
| no | no | no | – | NONE |
| yes | no | no/yes-but-diffuse | no | **QUALITY_DRIFT** (sensor/capture shift; known classes move together) |
| any | yes | no | no | **CONCEPT_DRIFT** (unexplained A-shift that is global: per-class shift vectors aligned, cosine > ρ) |
| any | yes or no | yes | yes | **NOVEL_ATTACK** (compact new cluster, far from every known class, not explained by Q, high M2 uncertainty) |
| yes | yes | yes | yes | **MIXED** (both are reported, novelty first) |

\*Coherent means HDBSCAN cluster size ≥ m_min, stability ≥ s_min, silhouette ≥ 0.25, and Mahalanobis distance to the nearest known-class centroid above the 99th reference percentile. Isolated outliers count as noise. Prior shift (more attacks of a *known* type) shows up as output drift with no H_novel rejection, and is labelled `OUTPUT_DRIFT/PRIOR_SHIFT`.

This extends the MINAS idea [84] (cluster the unknowns, then decide between "extension of a known concept" and "new concept") with two additions: (i) quality-conditional testing, so capture-condition changes are not mistaken for new attacks, and (ii) calibrated window-level p-values in place of heuristic thresholds. Section E7 of `EXPERIMENT_PLAN.md` evaluates it explicitly.

## 10. Recommendation engine

The **sensitivity matrix** `S[module, q_feature]` is estimated offline. Corruptions are applied at 5 severities to validation data, and S is the slope of module AUC (and ECE) against the corruption-induced change in each q feature (experiment E5a). When a QUALITY_DRIFT alert names feature f with effect size e, the recommender ranks modules by `S[m, f]·e` and emits a templated, evidence-backed message such as:

> Drift in `blur_laplacian_var` (PSI 0.41 [crit 0.08 @ n=500], KS p<1e-6 after BH, window 14:00–15:00, bundle v0.3.1). Estimated M2 AUC drop at the current blur level: −0.07 (95% CI −0.10…−0.04). Recommended action: augment M2 training with Gaussian/motion blur σ∈[1,3] and recalibrate.

Recommendations are validated in E8, which checks whether following them recovers performance better than random augmentation.

## 11. Frontend

- **/kyc:** consent screen (purpose, retention, rights, withdraw), ID upload (image/PDF; PDF first page via pdf.js), selfie recorder (MediaRecorder WebM, 6–8 s, randomized server-issued challenge script with timestamps), progress view, result view. The result view has the decision badge, per-module gauges with confidence, fusion weights, landmark overlay, rPPG plot, Grad-CAM/ELA panels, contributing features, and user-friendly reason text.
- **/monitor:** drift timeline, per-feature heatmap (feature × window, coloured by −log10 q-value), top drifting features with recommendations, novelty-score distributions (reference vs current), novel-cluster viewer, alert feed with evidence drill-down, and a control panel for window size/step, α, and alert thresholds.
- Accessibility: keyboard-navigable controls, ARIA labels on charts, colour plus text/icon on badges (not colour alone), captions for challenge prompts.

## 12. Security, privacy, compliance

- **Authentication.** The KYC flow uses a per-session token. The Dashboard 2 and admin APIs require an API key (env `ADMIN_API_KEY`). Without one they refuse to start, unless `DEMO_MODE=1`, which binds to localhost and shows a banner. **Demo mode is unauthenticated and must not be exposed to a network.**
- **Encryption.** Media and identity templates use AES-256-GCM (`cryptography`), with the key from env/secret file. Key rotation is supported through a key id in the file header.
- **Retention.** `MEDIA_TTL_DAYS` (default 30, configurable) applies to raw media and templates. Features and artifact embeddings follow `LOG_TTL_DAYS`. A purge job runs daily. `DELETE /sessions/{id}` gives erasure on request (DPDP Act [87] erasure; GDPR Art. 17). Regulated deployments may need *longer* retention of V-CIP recordings [86], so the setting is configurable and documented as a legal decision, not a technical one.
- **Logs.** Structured JSON logging (structlog) with no raw images, no names or ID numbers, and no face embeddings. OCR text is reduced to validity flags (check-digit pass/fail, field consistency) and never stored verbatim.
- **Notes.** GDPR Art. 9 treats biometric data used for identification as a special category, so explicit consent and a DPIA are needed. DPDP Act consent, notice and purpose limitation are covered in `docs/PRIVACY.md` (milestone 2).

## 13. Model versioning

A `ModelBundle` manifest (YAML) lists, for each module, the weights hash, calibrator hash, training data manifest hash and code git SHA, plus the fusion config, policy thresholds and reference window id. `bundle_version` is a semantic version plus a short hash. Every session, report and alert stores it. When the bundle changes, the monitor starts a new reference window and never compares across bundles without saying so.

## 14. Deployment

- `docker compose up`: `api` (python:3.11-slim, CPU torch wheels, MediaPipe, OpenCV-headless), `worker` (same image, runs the monitor scheduler and the inference process pool), `frontend` (Vite build served by nginx), and an optional `postgres` profile.
- `DEMO_MODE=1 CPU_ONLY=1` (default): small models (CLIP ViT-B/16, MediaPipe, buffalo_l or facenet), 16 frames, target < 15 s per session on a laptop CPU. This will be measured in E9, not promised.
- Local dev without Docker: Python 3.11 is preferred; the host has 3.12.10, which MediaPipe/torch/OpenCV support. The pinned `requirements.lock` is tested on 3.11 in CI.

## 15. Technology choices and justification

| Choice | Why |
|---|---|
| FastAPI + Pydantic | Typed schemas shared with the frontend through OpenAPI; async upload handling. |
| Process pool instead of Celery/Redis | One-host demo; avoids two extra services. The JobRunner interface allows swapping in Celery later. |
| SQLAlchemy + SQLite default | Zero-setup reproducibility; Postgres through one env var. |
| MediaPipe Face Landmarker (478 pts incl. iris) | Specified by the task; CPU real-time [43–45]; Apache-2.0. |
| open_clip CLIP ViT-B/16 frozen | UnivFD evidence [21]; CPU-feasible; the embedding doubles as the novelty space. |
| scikit-learn heads and HDBSCAN | CPU, deterministic with seeds, no extra deps (`sklearn.cluster.HDBSCAN`). |
| In-house drift/OOD tests + river/alibi-detect/evidently only as test oracles | Requirement: core tests must be controllable and auditable. |
| PyAV for decoding | Exposes bitrate, codec and frame timestamps needed for q. |

## 16. Known limitations (carried into the paper)

- Composite sessions (ID from SIDTD, selfie from FF++/Celeb-DF/OULU) are not real KYC captures. Correlations between document and selfie fraud are artificial.
- Challenge-response (M4) cannot be evaluated on public datasets, which have no challenges. It can only be evaluated on a small consented in-house recording set, and is otherwise reported as a qualitative demo.
- RGB webcam only; no depth/IR [31].
- Frozen CLIP features may encode identity and demographics. Novelty clustering in that space may group by demographic rather than by attack. This is audited in E10.
- Injection attacks (virtual cameras that bypass the physical camera) are out of scope for M3, which only addresses presentation attacks. This is listed as future work.


## 17. Decisions and deviations recorded in milestone 2

- **Training on Kaggle, inference on CPU.** All training runs as Kaggle notebooks generated from job YAMLs (`training/`, `docs/KAGGLE.md`). Every job has a CPU `smoke` mode and a Kaggle `full` mode. TruFor, CDCN and SBI-EfficientNet are Kaggle-only plug-ins. The served default stays frozen CLIP plus a linear head.
- **Data via Kaggle.** Logical dataset names map to Kaggle mirrors in `training/datasets.yaml`. The original licenses still apply. DocTamper and OULU-NPU are optional.
- **"Read digits" check.** Offline Vosk ASR (`codmav-weights-vosk`). If Vosk is missing, the check falls back to mouth-aperture activity in the digits window, and the output records which method was used. This lands in milestone 3.
- **Face matching.** InsightFace ArcFace (non-commercial license noted) behind the M5 interface, with facenet-pytorch as a drop-in swap.
- **Job runner.** A thread pool inside the API process (`JobRunner`), not a separate worker container. A `worker` service gets added in milestone 5, when the monitor scheduler exists.
- **Decision before calibration.** `config/policy.yaml` thresholds are `null`, so `decide()` returns `UNAVAILABLE` with a reason. It never applies guessed thresholds.
- **Upload validation.** Magic-byte sniffing (JPEG/PNG/WebP/PDF; WebM/MP4) plus size limits. Declared content types are not trusted.
- **Python 3.12 instead of 3.11.** The pinned, tested stack (numpy 2.5.4, scipy 1.18.1) requires Python ≥3.12, and the laptop and test environment run 3.12. The Docker image is `python:3.12-slim`. This was checked with a successful `docker compose build` and an end-to-end session through nginx (`scripts/e2e_smoke.py`).
- **Code provenance without git.** `package_src` stamps `GIT_SHA` as `<git sha>[-dirty]+src.<sha256 of the packaged files>`, or `nogit+src.<hash>` when there is no git repo. Every run manifest and bundle therefore points to the exact code that produced it.
