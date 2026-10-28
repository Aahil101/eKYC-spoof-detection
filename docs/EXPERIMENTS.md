# Experiments (milestone 6): consolidated results

Reproduce:
- **Kaggle jobs:** `training/configs/*.yaml`, notebooks `aahil1011/codmav-*`.
- **Local analysis:** with the Kaggle feature shards on `CODMAV_INPUT_ROOTS`:
  - `python -m experiments.paper_tables --heads <train_heads artifacts>` (E1, E3)
  - `python -m experiments.e2_e10 --heads … --fusion …` (E2, E10; writes `docs/figures/e2_reliability.png`)
  - `python -m experiments.e5_corruptions --heads … --fusion … --reference …` (E5)
  - `python -m experiments.e6_attribution --seeds 0 1 2` (E6b)
  - `python -m experiments.e8_closed_loop --seeds 0 1 2 --heads …` (E8)
  - `python -m experiments.e9_latency --videos a.mp4 b.mp4 --portrait face.png` (E9; needs a bundle)
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


## E2: calibration (10 equal-width bins; ECE with grouped-bootstrap 95% CI)

| model / test set | n | ECE [95% CI] | MCE | Brier |
|---|---|---|---|---|
| M2, FF++ test | 822 | 0.146 [0.119, 0.174] | 0.338 | 0.128 |
| M2, Celeb-DF test | 516 | 0.138 [0.109, 0.176] | 0.309 | 0.200 |
| M3, MSU-MFSD test | 159 | 0.105 [0.063, 0.164] | 0.643 | 0.112 |
| M3, CelebA-Spoof test | 5997 | 0.023 [0.018, 0.032] | 0.211 | 0.046 |
| M4, MSU-MFSD test | 159 | 0.125 [0.080, 0.190] | 0.473 | 0.180 |
| Fusion F3, composite fusion_test | 1335 | 0.057 [0.041, 0.077] | 0.606 | 0.059 |

Figure: `docs/figures/e2_reliability.png`. M2 overestimates P(genuine) across the whole range on FF++: every bin lies below the diagonal, e.g. predicted 0.85 → observed 0.67. This is the FF++ val→test calibration shift. The fused score is better calibrated than every video module, but its mid-range bins are sparse (most sessions sit near 0 or 1), so the MCE is not meaningful there.

## E6b: forgery-attribution embedding (leave-one-method-out, FF++; 3 seeds)
`ml/attribution.py`: a 2-layer MLP over the 541-d artifact vector, trained with cross-entropy + supervised contrastive loss on FF++ train classes without the held-out method. OOD score in the L2-normalised 64-d embedding.

| held-out method | closed-set acc | kNN AUROC | Maha AUROC | energy AUROC | kNN FPR@95 | control: Celeb-DF genuine (kNN) | PCA32 baseline (best scorer) |
|---|---|---|---|---|---|---|---|
| Deepfakes | 0.643 | 0.575 ± 0.018 | 0.556 ± 0.019 | 0.549 ± 0.011 | 0.914 | 0.545 ± 0.021 | 0.658 ± 0.002 |
| Face2Face | 0.688 | 0.641 ± 0.016 | 0.653 ± 0.009 | 0.697 ± 0.022 | 0.787 | 0.619 ± 0.020 | 0.536 ± 0.001 |
| FaceSwap | 0.641 | 0.662 ± 0.005 | 0.645 ± 0.002 | 0.577 ± 0.014 | 0.852 | 0.619 ± 0.029 | 0.558 ± 0.004 |
| NeuralTextures | 0.710 | 0.638 ± 0.008 | 0.647 ± 0.012 | 0.655 ± 0.012 | 0.748 | 0.691 ± 0.020 | 0.581 ± 0.000 |
| FaceShifter | 0.655 | 0.679 ± 0.017 | 0.688 ± 0.020 | 0.652 ± 0.020 | 0.764 | 0.591 ± 0.026 | 0.546 ± 0.003 |
| **mean** | | **0.639** | 0.638 | 0.626 | | | 0.576 |

Adding the LBP histogram changes nothing (kNN mean 0.642, 1 seed).

**In the monitor** (`build_reference`; the attribution embedder is trained on head-training partitions only, so it is disjoint from both the reference and the stream):
- The leave-one-known-class-out selection on the reference still prefers pca128/LOF (0.575 vs supcon 0.539). Reference classes include replay attacks, whose features differ from FF++.
- Forcing `novelty_space: supcon64/knn`:
  - sample-level FaceShifter AUROC (Mahalanobis) 0.414 → 0.583;
  - the Celeb-DF genuine control 0.784 → 0.539, so the capture domain is no longer mistaken for an attack;
  - the OOD-only baseline's false novelty under domain drift 0.95 → 0.35.
- The proposed rule: macro 0.458, still 0% false novelty, still 0% novel detection.

**Conclusion:** supervised attribution moves the representation in the right direction and removes most of the domain confound. Held-out methods are still not separable enough (FPR@95 ≥ 0.75) for window-level detection. This is a weak positive at sample level and a negative at window level.

## E8: closed loop (does acting on the recommendation help?)
FF++ test subjects (the corrupted subset) are split 50/50 by subject hash. Half A is added to M2 training: clean copies, plus corrupted copies with the recommended corruption. Half B is the evaluation set. 3 seeds, AUC on B.

| condition | base (deployed recipe) | + A clean (control: more data) | + A recommended corruption | + A all 5 corruptions | recovery of lost AUC |
|---|---|---|---|---|---|
| clean | 0.902 ± 0.007 | 0.903 ± 0.012 | 0.902 ± 0.014 (on clean) | 0.897 ± 0.015 | |
| blur | 0.746 ± 0.009 | 0.743 ± 0.006 | **0.785 ± 0.001** | 0.781 ± 0.013 | 0.25 ± 0.05 |
| downscale | 0.779 ± 0.010 | 0.775 ± 0.006 | **0.800 ± 0.009** | 0.811 ± 0.016 | 0.17 ± 0.05 |
| JPEG | 0.716 ± 0.009 | 0.709 ± 0.010 | **0.759 ± 0.006** | 0.754 ± 0.012 | 0.23 ± 0.02 |
| low light | 0.667 ± 0.011 | 0.659 ± 0.012 | **0.736 ± 0.009** | 0.731 ± 0.018 | 0.29 ± 0.06 |
| noise | 0.768 ± 0.009 | 0.766 ± 0.006 | **0.795 ± 0.004** | 0.789 ± 0.008 | 0.20 ± 0.06 |

Recommendation-driven augmentation recovers 17–29% of the lost AUC on unseen subjects and does not cost clean accuracy. The more-data control gives no gain, so the effect comes from the corruption itself. Recovery is partial because the M2 backbone (CLIP) is frozen; only the probe is retrained.

**E8b, fusion under corruption** (M2–M4 + quality; M1/M5 absent; fitted on clean FF++ val):

| condition | F1 sum rule | F3 GBM stacking | F4 quality-gated |
|---|---|---|---|
| clean | **0.904 ± 0.005** | 0.790 ± 0.014 | 0.853 ± 0.034 |
| blur | **0.755 ± 0.011** | 0.631 ± 0.003 | 0.708 ± 0.021 |
| downscale | **0.783 ± 0.010** | 0.656 ± 0.018 | 0.743 ± 0.024 |
| JPEG | **0.720 ± 0.013** | 0.619 ± 0.022 | 0.669 ± 0.006 |
| low light | **0.668 ± 0.012** | 0.647 ± 0.008 | 0.634 ± 0.031 |
| noise | **0.770 ± 0.009** | 0.722 ± 0.012 | 0.700 ± 0.028 |

Quality gating does not degrade more gracefully than the fixed rule; this is a negative result for the "quality-aware fusion" hypothesis. In this video-only setting F3 overfits the small val set (840 sessions, 3 modules). The composite-session F3 result (E3) does not transfer to it.

## E9: latency
Intel CPU (Family 6 Model 140), 8 logical CPUs, no GPU; bundle kyc-v0.4; 6 steady-state sessions of 10–12 s FF++ videos.
- End-to-end `/submit`: median **11.4 s**, max 11.6 s. Warm-up (model loading): 14.9 s.
- Stored module latency includes 1/5 of the shared media analysis each (~1.9 s: decode, face tracking, CLIP/LBP/landmarks). M2's own CLIP cost adds about 0.9 s on top.

## E10: subgroup audit (capture conditions; operating point p ≥ 0.5 = genuine)
None of the datasets as extracted carries demographic labels, so **no demographic fairness audit was possible**. The audit covers the capture conditions that are annotated or measured.

| subgroup | BPCER (genuine rejected) | APCER (attacks accepted) |
|---|---|---|
| M3, MSU-MFSD, Android | 0.150 | 0.167 |
| M3, MSU-MFSD, laptop | **0.450** | 0.034 |
| M4, MSU-MFSD, Android / laptop | 0.650 / 0.700 | 0.133 / 0.119 |
| M2, FF++, blurriest tertile | **0.311** | 0.114 |
| M2, FF++, sharpest tertile | 0.149 | 0.185 |
| M2, FF++, darkest / brightest tertile | 0.261 / 0.217 | 0.167 / 0.149 |
| M3, CelebA-Spoof, spoof illumination codes 1–4 | (live images: 0.145) | 0.017–0.040 |

The deployed fusion policy sends genuine sessions to human review at very different rates by source: MSU-MFSD 97.5%, FF++ 41.6%, Celeb-DF 25.4%. Rejects are 0.0% / 0.7% / 4.0%. Users on low-quality laptop webcams, or in blurry captures, bear most of the friction, which is a fairness-relevant capture effect. Sample sizes are small (MSU: 20 genuine per device), so these are indicative only.

## Not done (stated in the paper)
- Demographic audit (no labels).
- DFDC, SadTalker and SimSwap as additional unseen generators.
- End-to-end fine-tuning of a forgery backbone for novelty (E6b uses frozen features).
- Latency on GPU and in the Docker image.
