# Modules M1–M5 (milestone 3)

Every module follows the same contract: `predict(session) -> ModuleOutput(score = calibrated P(genuine), confidence, features, explanations)`. Features are computed once per session by `ml.modules.analyse_session`, using the **same functions** as the Kaggle extraction jobs (`ml/features/*`), so training and serving features cannot diverge (`FEATURE_VERSION = fv1`). Heads are tabular models trained on Kaggle and loaded from a hash-verified bundle. A module without a head in the bundle stays `placeholder`. A module without usable input (no face, no ID portrait) returns `not_applicable` and never a guessed score.

## Shared face pipeline (`ml/common/face.py`)
- MediaPipe Face Landmarker in VIDEO mode: 478 landmarks including iris, blendshapes, and facial transformation matrix → yaw/pitch/roll.
- 16 evenly spaced frames feed M2, M3 and M5. A dense strip (≤8 s at 15 fps, 320 px) feeds rPPG, blink and motion.
- Crops: 224 px square with 25% margin (CLIP / texture), and a 112 px ArcFace crop aligned to the InsightFace 5-point template.
- The quality vector now has real face size and head pose instead of NaN.
- Measured on the laptop CPU: about 40 ms per frame for MediaPipe. A full session takes 13–18 s on CPU, with CLIP the largest part.

## Per-module design

| | Inputs | Features (`fv1`) | Head | Training data (Kaggle) | Explanation |
|---|---|---|---|---|---|
| **M1** document | ID image | ELA at q95/90/75 (mean, p99, block CV, block max-z); DCT histogram periodicity and blockiness; block noise-residual CV / max-z / kurtosis; glare; FFT moiré peaks; 24-bin azimuthal spectrum; MRZ check digits (ICAO 9303) when MRZ text is supplied | HistGB + isotonic | SIDTD templates (CC-BY-4.0), template-disjoint split | ELA map, noise-residual map |
| **M2** deepfake | face crops | frozen CLIP ViT-B/16 (OpenAI weights, QuickGELU) mean embedding (512); CLIP temporal step / consistency; azimuthal spectrum (24); landmark jitter | Logistic regression (linear probe, UnivFD [21]) + temperature scaling | FF++ c23 official splits: real vs Deepfakes / Face2Face / FaceSwap / NeuralTextures / FaceShifter. Cross-dataset: Celeb-DF v2 test list | Grad-CAM on the last CLIP block w.r.t. the probe logit |
| **M3** replay/PAD | face crops + frames | IDA (specular, blur, chromatic moments, colour diversity); moiré peaks (face and full frame); colour LBP HSV+YCrCb (354). Face-vs-background flow correlation is computed for video but **absent from image training data**, so the head ignores it | HistGB + isotonic | CelebA-Spoof subset (1,500 per spoof type, 4,500 live, 6,000 test, official intra-test labels). Cross-dataset: MSU-MFSD video | top feature deviations |
| **M4** liveness | dense strip, challenge script, audio | rPPG (POS and CHROM) SNR, HR, inter-ROI coherence, peak ratio; EAR blink count/rate; yaw/pitch range; landmark jitter; non-rigid motion (Procrustes residual) | Logistic regression + temperature | MSU-MFSD (official 15/20 subject protocol) | rPPG waveform, challenge timeline |
| M4 challenge | as above | per-step pass/fail: head turn ≥15° vs baseline, pitch ≥10°, 2 blinks (hysteresis), smile blendshape ≥0.4, digits via **Vosk** (digit-word grammar) or a **mouth-motion fallback** when there is no audio or Vosk | **Rule-based** (no public data). A failed challenge sets `challenge_failed`, which the decision policy treats as a hard rule (milestone 4) | — | step list with evidence |
| **M5** biometrics | ID portrait + frontal frames | ArcFace R50 cosine (mean/max/min) between the ID portrait and video, video self-consistency; 8 landmark ratios (IPD/face width, iris/IPD, jaw/face length, …), their CV across frontal frames, and \|ID−video\| deltas | Logistic regression + temperature | Celeb-DF v2 real videos: same-identity vs other-identity pairs (identity-disjoint split) | ID portrait crop, landmark overlay |

**Confidence** is `(1 − binary entropy of p) × in-distribution factor`. The factor comes from the kNN distance of the standardised feature vector to the training set, scaled by calibration-split percentiles. The test set reports `confidence_auroc_correct`, i.e. whether confidence predicts correctness, so this definition is measured rather than assumed to work.

**Storage per session:** artifact embeddings (`m2_clip`, `m3_lbp`, `m1_spectrum`) go into `embeddings` for novelty detection. The mean ArcFace embedding is stored **encrypted** in `identity_templates` and purged with the media. Explanations hold landmark points and small heatmaps, not raw frames.

## Third-party weights (`python -m ml.common.weights fetch`, hashes in `models/store/weights.lock.json`)
| key | source | license |
|---|---|---|
| mediapipe_landmarker | storage.googleapis.com/mediapipe-models (face_landmarker.task) | Apache-2.0 |
| arcface_buffalo_l | InsightFace v0.7 release `buffalo_l.zip` → `w600k_r50.onnx` | **non-commercial research only** |
| vosk_small_en | alphacephei.com vosk-model-small-en-us-0.15 | listed as Apache-2.0 on the model page (to verify) |
| CLIP ViT-B/16 openai | open_clip / Hugging Face hub cache | OpenAI CLIP model card (to verify) |

## Results (bundle `kyc-v0.3`, Kaggle run `codmav-train-heads` v1 (all five heads), feature version fv1, seed 0)

Single run with no CIs yet; bootstrap CIs and seeds come in E1. Score = P(genuine). APCER is measured at BPCER = 5%.

| Module | Test set | n (genuine / attack) | AUC | EER | ECE (cal → uncal) | APCER @ BPCER 5% | Conf. AUROC (correct) |
|---|---|---|---|---|---|---|---|
| M2 deepfake | FF++ c23 official test | 140 / 700 | **0.871** | 0.201 | 0.150 / 0.129 | 0.484 | 0.728 |
| M2 cross-dataset | Celeb-DF v2 test list (unseen) | 178 / 340 | **0.769** | 0.303 | 0.144 | 0.797 | — |
| M2 per method (AUC) | FF++ test | | DF 0.94 · F2F 0.85 · FS 0.91 · NT 0.78 · FaceShifter 0.87 | | | | |
| M4 liveness | MSU-MFSD official test (20 subjects) | 40 / 119 | **0.691** | 0.347 | 0.116 | 0.815 | 0.702 |
| M5 face match | Celeb-DF identity pairs, identity-disjoint | 102 / 306 | **0.995** | 0.010 | 0.005 | 0.000 (FAR) | 0.649 |
| M1 document | SIDTD, template-disjoint | 185 / 239 | 1.000 ⚠ | 0.000 | 0.000 | 0.000 | — |
| M3 replay | CelebA-Spoof intra-test (images) | 1,796 / 4,201 | **0.989** | 0.047 | 0.026 / 0.025 | 0.121 | 0.796 |
| M3 cross-dataset | MSU-MFSD all 278 videos (unseen) | 70 / 208 | **0.430** ⚠ | 0.550 | 0.287 | 1.000 | — |

Honest reading:
- **M1 = 1.000 is almost certainly a dataset shortcut, not real forgery skill.** SIDTD fakes differ from reals in global compression: ELA q95 mean is 2.38 for reals vs 0.68 for fakes, and DCT blockiness 1.03 vs 1.15, because the forgeries were re-encoded. The model separates "re-saved JPEG" from "original", so the score must not be reported as forgery detection. On a synthetic test document the live API gives M1 = 0.00 with confidence 0.00 (out of distribution). **TODO (E1): retrain with quality-matched re-encoding of reals, evaluate cross-dataset, report both.**
- **M3 does not transfer from images to video (cross-dataset AUC 0.43, worse than chance).** On the live check it gives a **genuine** FF++ video 0.00 with confidence 0.96. That is a confident false reject, and the kNN confidence does not catch it because the frame-texture features fall inside the CelebA-Spoof training range. This image→video domain gap (CelebA-Spoof still photos vs compressed video frames) is a real limitation, and it also shows the confidence measure can fail. **Fix planned for the start of milestone 4:** retrain M3 on CelebA-Spoof plus the MSU-MFSD *train* subjects (video domain), evaluate on the MSU *test* subjects, and keep M3 out of the hard veto until cross-domain AUC is acceptable.
- **M2 is weakest on NeuralTextures and loses about 0.10 AUC cross-dataset**, consistent with the literature. Calibration got slightly *worse* after temperature scaling on test (0.129 → 0.150). The calibration split is FF++ val, and this shows exactly the calibration shift that motivates monitoring.
- **M4 is weak**, as expected from MSU-MFSD's size and compression.
- **M5's near-perfect result is on Celeb-DF video-vs-video pairs.** Printed or scanned ID portraits will be harder (domain gap, see limitations).

Live API check (`kyc-v0.3`, laptop CPU): FF++ original 000 → M2 0.85, M5 1.00 against its own face on the ID. FF++ Deepfakes 000_003 (face of 003 swapped onto 000) → **M2 0.09, M5 0.015**: both modules flag the fake. 15–37 s per session on CPU.

## Known limitations (also for the paper)
- **Blink rate depends on content.** On FF++ news-anchor clips (people talking) EAR gives about 45 counts/min. On MSU-MFSD (people sitting still) live faces show a median of 15/min, which is plausible physiologically, against 7.5/min for attacks. Talking inflates the count, so `blink_rate_per_min` is an evidence feature, not a physiological measurement.
- **rPPG separates little on these datasets.** MSU-MFSD medians are POS SNR −2.0 dB live vs −1.7 dB attack. It is a weak cue on compressed video, consistent with the literature, and is kept for the ablation (does rPPG help at all?).
- **Two MSU-MFSD clips are undecodable** in the Kaggle mirror (`attack_client008_laptop_SD_ipad_video`, `attack_client023_laptop_SD_iphone_video`; FFmpeg "Invalid data"). They are excluded and reported.
- **M3 is trained on still images** (CelebA-Spoof), so temporal cues (flow correlation) are unused. Cross-dataset results on MSU-MFSD video measure the image→video gap.
- **M4 has little data:** MSU-MFSD has 35 subjects and 280 clips. Expect wide confidence intervals.
- **M5 ID portraits are proxies:** in training, the "ID photo" is a frame from a different video of the same person, not a printed ID portrait (no public paired ID+selfie data exists). Domain gap: print/scan of a real ID.
- **M1 is not cross-dataset validated:** SIDTD forgeries (crop-and-replace, inpainting) are one forgery family. Robustness to recapture and AI-generated IDs is not shown.
- **No SBI pseudo-fakes yet** (planned plug-in). M2 is trained only on FF++ manipulations.
