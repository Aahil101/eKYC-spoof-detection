# Monitoring: drift, novel attacks and drift-vs-novelty (milestone 5)

Code: `monitoring/drift.py`, `monitoring/novelty.py`, `monitoring/disambiguation.py`, `monitoring/recommend.py`, `monitoring/service.py`. Kaggle job: `training/monitor_job.py` → `aahil1011/codmav-build-reference` v1. Served in bundle **`kyc-v0.4` = 0.4.0+a0a464f8da** (`MONITOR/reference_served.pkl`). API: `/api/monitor/*` (admin). UI: `/monitor` (Dashboard 2).

## What is implemented
- **Per-feature drift:** KS, Wasserstein-1 (in reference-SD units), PSI with a sample-size-aware χ² critical value [66], Jensen–Shannon; Benjamini–Hochberg across features.
- **Multivariate drift:** MMD² (RBF, median heuristic, permutation p) and C2ST (binomial null, permutation importances for attribution). Density-ratio weights w(q) come from a classifier [64].
- **Streaming:** ADWIN, Page–Hinkley (fused-score stream), DDM, EDDM (error stream from human-review labels).
- **Output drift:** χ² test on decision rates, KS test on fused scores.
- **Novelty:**
  - Scorers on artifact embeddings (CLIP face embedding + colour-LBP; never identity): class-conditional Mahalanobis, kNN, Isolation Forest, LOF.
  - Every score becomes a conformal p-value.
  - Novel-candidate clusters use DBSCAN with eps = the reference within-class kNN radius (the cluster must be as tight as a known attack family). sklearn HDBSCAN was tried first: its default discards a single new cluster, and even with `allow_single_cluster` it recovered 8 of 40 points of a tight synthetic cluster, so it is not used.
  - The space and scorer are chosen automatically by **leave-one-known-class-out on the reference only** (no test data).
- **Drift-vs-novelty rule** (ARCHITECTURE §9): H_Q, H_novel, H_A|Q on inliers with quality reweighting, plus coherent-cluster → {NONE, QUALITY_DRIFT, CONCEPT_DRIFT, NOVEL_ATTACK, MIXED, OUTPUT_DRIFT}, with a confidence value.
- **Recommendations:** templated, evidence-backed text naming the feature, statistics, window, bundle and the affected module, from an **observational** sensitivity table (module AUC in reference tertiles). The interventional E5a table replaces it later.
- **Alerts:** severity by label, stored with evidence and the bundle version. Reports store per-feature statistics, histograms and cluster members.

## A methodological finding: iid tests are invalid for KYC streams
The first version flagged **every** window as CONCEPT_DRIFT, including the stationary one. Two causes:
1. Exact duplicate artifact vectors: a doc-forgery session reuses its genuine video, and sampling was done with replacement. Duplicates inflate MMD through off-diagonal k = 1 terms.
2. More fundamentally, **sessions are clustered by person** (FF++ has 6 videos per subject). A window of new people differs from the reference people, and iid permutation/binomial tests treat that as drift.

Fix: a **subject-level empirical null**. Monitor statistics are computed on windows drawn from burn-in subjects (in production, the first weeks after deployment), and every window p-value is the empirical rank of the observed statistic in that null. Windows never reuse burn-in subjects. This belongs in the paper's method section; it is why naive two-sample monitoring raises false alarms on biometric streams.

## Pilot results on real held-out features
Reference: FF++/MSU-MFSD `fusion_train` (696 videos). **Celeb-DF is never in the reference**, so its real videos are an unseen capture domain and its fakes an unseen generator (M2 never saw Celeb-DF). Novel-attack hold-out: FF++ **FaceShifter**, removed from the reference. Window 200 sessions, 20 repeats, null from 60 burn-in windows (62 burn-in / 95 evaluation subjects). The pilot runs on composite sessions from held-out features (no media corruption yet; that is E5).

| Scenario | Expected label | Correct | H_Q fired | H_novel fired | coherent cluster |
|---|---|---|---|---|---|
| S0 stationary (new subjects, same mix) | NONE | **90%** (10% false alarms) | 10% | 25% | 0% |
| S1 domain drift (Celeb-DF genuine replace FF++/MSU genuine) | QUALITY_/CONCEPT_DRIFT | **100%** | 100% | 25% | 0% |
| S2 novel attack (15% FaceShifter, held out) | NOVEL_ATTACK | **0%** | 15% | 5% | 0% |
| S3 mixed (S1 + 15% FaceShifter) | MIXED | **0%** (reported as QUALITY_DRIFT) | 95% | 10% | 0% |
| S4 prior shift (30% Face2Face, a known class) | OUTPUT_DRIFT (not novelty) | **80%** | 15% | 15% | 0% |
| S5 unseen generator + new domain (15% Celeb-DF fakes) | NOVEL_ATTACK/MIXED | **0%** (reported as QUALITY_DRIFT) | 95% | 10% | 0% |

Macro accuracy over the six scenarios: **0.45**.

Sample-level novelty (AUROC, held-out vs known-class test sessions): FaceShifter 0.38–0.57 (FPR@95TPR ≥ 0.91); Celeb-DF fakes 0.67–0.80, confounded with the Celeb-DF capture domain. Leave-one-known-class-out on the reference selected PCA-128 + LOF, with a mean AUROC of only 0.60 across candidates.

**Honest conclusion:**
- **Drift detection works.** False alarms are 10% at α = 0.05 after the subject-level null (iid tests: 100%), domain drift is detected in every window, and prior shift is correctly *not* called novelty (80%).
- **Novel-attack detection does not work in this artifact space.** A held-out FF++ manipulation method is *not* out-of-distribution in frozen CLIP + LBP features; its samples sit inside the known deepfake classes. The novelty half of the drift-vs-novelty claim is therefore **not supported yet**.
- The cause is the representation, not the decision rule. On synthetic data where the novel class is separable, the rule labels NOVEL_ATTACK and MIXED correctly (`tests/test_m5_monitoring.py`).

**Next step (E6, planned):** compute novelty in a **forgery-specific** space rather than a generic image space. Candidates: the M2 linear probe's discriminative subspace, a supervised-contrastive head trained on known methods, or DF40-style generator-attribution features. Then re-run S2/S3/S5 unchanged. SadTalker/SimSwap from the Kaggle DF40 slice are the next unseen generators.

## Live API check
Simulated streams via `POST /api/monitor/simulate` (held-out features, no media) followed by `POST /api/monitor/run`:
- stationary → **NONE**
- domain_drift → **QUALITY_DRIFT** (medium alert). Recommendation: *"Drift detected in blur_laplacian_var (PSI 0.31 [crit 0.105], KS q=3.9e-02, shift −0.48 SD …). M5 AUC on reference sessions with low blur_laplacian_var is 0.966 (−0.011 vs overall; observational) … augment training with Gaussian/motion blur …"*
- unseen_generator → QUALITY_DRIFT (the novelty is missed, consistent with the pilot)
