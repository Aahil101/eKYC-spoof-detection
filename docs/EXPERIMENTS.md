# Experiments (milestone 6): consolidated results

Reproduce:
- **Kaggle jobs:** `training/configs/*.yaml`, notebooks `aahil1011/codmav-*`.
- **Local analysis:** with the Kaggle feature shards on `CODMAV_INPUT_ROOTS`, run `python -m experiments.paper_tables --heads <train_heads artifacts>` (E1, E3) and `python -m experiments.e5_corruptions --heads … --fusion … --reference …` (E5).
- **Outputs:** `experiments/results/` (git-ignored); the tables below are copied from it.
- **Seeds:** 0–4 for E3, 0 elsewhere. CIs come from a grouped bootstrap (subjects, templates or identities resampled), 500–1000 replicates.

## E1: modules (point estimate [95% CI])

| module / test set | n (genuine) | AUC | EER | ECE | APCER @ BPCER 5% |
|---|---|---|---|---|---|
| M2 deepfake, FF++ test | 840 (140) | 0.871 [0.849, 0.896] | 0.201 [0.170, 0.236] | 0.150 [0.121, 0.178] | 0.484 [0.361, 0.605] |
| M2 deepfake, Celeb-DF test (cross-dataset) | 518 (178) | 0.769 [0.735, 0.806] | 0.303 [0.264, 0.331] | 0.144 [0.119, 0.184] | 0.797 [0.671, 0.865] |
| M3 replay, MSU-MFSD test (video) | 159 (40) | 0.914 [0.853, 0.962] | 0.172 [0.109, 0.293] | 0.112 [0.070, 0.166] | 0.462 [0.185, 0.542] |
| M3 replay, CelebA-Spoof test (images) | 5997 (1796) | 0.979 [0.975, 0.983] | 0.071 [0.062, 0.081] | 0.024 [0.019, 0.032] | 0.096 [0.079, 0.250] |
| M4 liveness, MSU-MFSD test | 159 (40) | 0.691 [0.586, 0.788] | 0.347 [0.275, 0.443] | 0.116 [0.091, 0.206] | 0.815 [0.667, 0.958] |
| M1 document, SIDTD test (seen designs) | 420 (185) | 1.000 [1.000, 1.000] | 0.000 | 0.000 | 0.000 |
| M1 document, **leave-one-document-type-out** | 10 folds | **0.548** (min 0.121, max 1.000) | | | |
| M5 face match, Celeb-DF identity pairs | 408 (102) | 0.995 [0.985, 1.000] | 0.010 [0.000, 0.063] | 0.005 [0.003, 0.015] | 0.000 [0.000, 0.170] |

## E3: fusion (fusion_test, mean ± sd over 5 seeds)

| strategy | AUC | EER | ECE |
|---|---|---|---|
| F0 mean | 0.868 ± 0.005 | 0.193 ± 0.013 | 0.047 ± 0.003 |
| F1 fixed sum rule | 0.931 ± 0.004 | 0.153 ± 0.005 | 0.039 ± 0.003 |
| F2 LR stacking | 0.956 ± 0.005 | 0.110 ± 0.007 | 0.034 ± 0.001 |
| **F3 GBM stacking + quality** | **0.963 ± 0.005** | **0.092 ± 0.010** | 0.069 ± 0.009 |
| F4 quality-gated mixture | 0.917 ± 0.011 | 0.161 ± 0.012 | 0.061 ± 0.018 |
| F5 Dempster–Shafer | 0.725 ± 0.004 | 0.298 ± 0.003 | 0.043 ± 0.011 |
| F6 DS + gated discount | 0.878 ± 0.022 | 0.198 ± 0.018 | 0.066 ± 0.015 |

Paired bootstrap of AUC differences (seed 0): **F3−F1 +0.036 [+0.026, +0.047]** (significant); F3−F2 +0.006 [−0.003, +0.013] (n.s.); **F4−F1 −0.005 [−0.020, +0.011] (n.s.: gating does not help on clean data)**; F6−F5 +0.180 [+0.154, +0.208].

## E4: policy
See `docs/FUSION.md`. The worst-type APCER target of 1% gives 5.1% on test (FaceShifter). Genuine sessions: 55.8% accepted, 42.2% to review, 2.0% rejected.

## E5: controlled corruptions (FF++ test subset: 140 real + 40 per method; severity 2; paired clean vs corrupted)

**E5a, interventional sensitivity (ΔAUC = corrupted − clean, same videos):**

| corruption | n | M2 ΔAUC | M3 ΔAUC | M4 ΔAUC | largest quality shift (ref SD) |
|---|---|---|---|---|---|
| blur σ=2 | 332 | **−0.173** | +0.008 | +0.009 | blur_laplacian_var −2.48 |
| downscale ×0.4 | 330 | **−0.139** | +0.008 | −0.027 | blur_laplacian_var −2.09 |
| JPEG q=20 | 331 | **−0.170** | −0.010 | −0.005 | noise_sigma −1.79 |
| low light (gain 0.4) | 331 | **−0.238** | +0.016 | −0.044 | noise_sigma +10.78 |
| Gaussian noise σ=12 | 335 | **−0.133** | +0.012 | −0.030 | noise_sigma +15.73 |

**The deepfake detector (frozen CLIP probe) loses 0.13–0.24 AUC under every common capture degradation.** Texture PAD (M3) and liveness (M4) are largely unaffected. This is the interventional evidence behind the recommendation *"fine-tune / augment M2 with <condition>"*.

**E5b, drift detection on windows of 200 corrupted sessions** (served reference + subject-level null, 20 windows each):

| corruption | detection | false-novelty | attribution hit@1 | hit@3 | alarm rate, same videos uncorrupted |
|---|---|---|---|---|---|
| blur | **1.00** | 0.00 | 1.00 | 1.00 | **0.00** |
| downscale | 1.00 | 0.30 | 1.00 | 1.00 | 0.00 |
| JPEG | 1.00 | 0.40 | 1.00 | 1.00 | 0.00 |
| low light | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 |
| noise | 1.00 | 0.00 | 1.00 | 1.00 | 0.00 |

Every corruption is detected, with no input-level alarms on the same videos uncorrupted, and the corrupted quality feature is in the top 3 every time (top 1 in 4 of 5). Remaining weakness: downscale and JPEG are mislabelled MIXED (a false new attack) in 30–40% of windows, because these corruptions change the forgery features more than the quality-reweighting model explains. Two hardening steps from this experiment: novelty must not be quality-explained, and outliers must be a minority.

## E6: novel attacks (leave-one-method-out) — negative result
Held-out FaceShifter is **not out-of-distribution** in any representation tried (AUROC 0.39–0.57):
- generic CLIP+LBP;
- the forgery-specific vector (CLIP + spectrum + temporal M2 cues);
- PCA and supervised LDA spaces;
- Mahalanobis, kNN, Isolation Forest, LOF, energy and MSP scorers.

LDA spaces were also rejected because they overfit the reference (H_A|Q fired in 70% of stationary windows). Celeb-DF fakes score 0.70–0.92, but **Celeb-DF genuine videos score 0.76–0.86**. That signal is the capture domain, not forgery novelty.

## E7: drift vs novelty (6 scenarios × 20 windows, real features)

| scenario | proposed | false novelty | OOD-rate-only baseline (false novelty) | proposed without quality conditioning |
|---|---|---|---|---|
| S0 stationary | 0.90 | 0.00 | 0.70 (0.25) | 0.90 |
| S1 domain drift | 1.00 | 0.00 | 0.05 (**0.95**) | 1.00 |
| S2 novel attack (FaceShifter) | 0.00 | 0.00 | 0.25 (0.25) | 0.00 |
| S3 mixed | 0.00 | 0.00 | 0.00 (0.85) | 0.00 |
| S4 prior shift (known class) | 0.80 | 0.00 | 0.60 (0.30) | 0.95 |
| S5 unseen generator + domain | 0.00 | 0.00 | 0.80 (0.80) | 0.00 |
| macro | **0.45** | **0.00** | 0.40 | 0.475 |

The proposed rule never calls drift a new attack. The OOD-only monitor does so in 95% of domain-drift windows. Detecting real novelty fails, for the representation reason given in E6.

## E7b: null model
Stationary windows flagged with iid tests: **100%**. With the subject-level empirical null: **10%**.

## Not done (stated in the paper)
- E2 reliability diagrams (ECE values only).
- E8 closed-loop retraining (needs fine-tuning M2 on corrupted data).
- E9 is only a laptop CPU measurement.
- E10 bias audit.
- DFDC, SadTalker and SimSwap as additional unseen generators.
