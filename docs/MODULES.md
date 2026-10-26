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

## Known limitations (also for the paper)
- **Blink rate depends on content.** On FF++ news-anchor clips (people talking) EAR gives about 45 counts/min. On MSU-MFSD (people sitting still) live faces show a median of 15/min, which is plausible physiologically, against 7.5/min for attacks. Talking inflates the count, so `blink_rate_per_min` is an evidence feature, not a physiological measurement.
- **rPPG separates little on these datasets.** MSU-MFSD medians are POS SNR −2.0 dB live vs −1.7 dB attack. It is a weak cue on compressed video, consistent with the literature, and is kept for the ablation (does rPPG help at all?).
- **Two MSU-MFSD clips are undecodable** in the Kaggle mirror (`attack_client008_laptop_SD_ipad_video`, `attack_client023_laptop_SD_iphone_video`; FFmpeg "Invalid data"). They are excluded and reported.
- **M3 is trained on still images** (CelebA-Spoof), so temporal cues (flow correlation) are unused. Cross-dataset results on MSU-MFSD video measure the image→video gap.
- **M4 has little data:** MSU-MFSD has 35 subjects and 280 clips. Expect wide confidence intervals.
- **M5 ID portraits are proxies:** in training, the "ID photo" is a frame from a different video of the same person, not a printed ID portrait (no public paired ID+selfie data exists). Domain gap: print/scan of a real ID.
- **M1 is not cross-dataset validated:** SIDTD forgeries (crop-and-replace, inpainting) are one forgery family. Robustness to recapture and AI-generated IDs is not shown.
- **No SBI pseudo-fakes yet** (planned plug-in). M2 is trained only on FF++ manipulations.
