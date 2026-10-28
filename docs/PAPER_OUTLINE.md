# Paper outline (draft, milestone 6)

Working title: **"Telling capture drift from new attacks: honest monitoring of a multi-module video-KYC system"**

Every number below comes from a run in this repository; the "Evidence" column names the file that produced it. Bracketed citations refer to `docs/RESEARCH_NOTES.md`. Only papers verified there are cited.

## 1. Problem statement
Remote video-KYC combines document forensics, deepfake detection, presentation-attack detection, liveness and face matching. Deployed systems face two kinds of change: **capture drift** (new phones, compression, lighting), which degrades modules silently, and **novel attacks** (new generators), which known-attack detectors miss [18, 21, 37]. Monitoring tools test for distribution shift [60–62] or flag out-of-distribution samples [72–81], but do not tell an operator *which* of the two is happening. They are also not validated on clustered biometric streams. We build an end-to-end, reproducible system and evaluate it honestly, including where it fails.

## 2. Related work (from RESEARCH_NOTES)
- Document forensics [1–13]
- Deepfake detection and generalisation [14–26]
- Presentation attack detection [27–37]
- rPPG and liveness [38–42]
- Facial geometry and face recognition [43–46]
- Quality-aware and evidential fusion [49–58]
- Drift detection [59–71]
- OOD and open-set detection, conformal outlier testing, novelty in streams [72–84]
- MINAS [84] is the closest prior work on separating concept evolution from drift.

## 3. Method
1. **Modules M1–M5** with a common interface and calibrated outputs (temperature or isotonic scaling [54, 55]). Frozen CLIP + linear probe [21] for M2. Texture/moiré PAD [33–35] for M3. POS/CHROM rPPG [38, 39] + EAR [42] + challenge-response for M4. ArcFace [46] + landmark ratios for M5. ELA/DCT/noise forensics [1–3] for M1. kNN-based confidence (`docs/MODULES.md`).
2. **Composite sessions** from held-out partitions; fusion F0–F6 including quality-gated [50, 53] and Dempster–Shafer [58] variants (`docs/FUSION.md`).
3. **Data-driven decision policy:** worst-case-per-attack-type APCER target, out-of-fold thresholds, and a veto certified by the rule of three.
4. **Monitoring:** per-feature and multivariate tests [60–66], streaming detectors [67–70], conformal novelty [81]. Two key ingredients:
   - (a) a **subject-level empirical null**, needed because biometric sessions are clustered by person;
   - (b) a **quality-conditional drift-vs-novelty rule**: novelty is declared only for a coherent outlier cluster that is not explained by a quality shift (density-ratio reweighting [64]); cf. MINAS [84].
5. **Recommendations** come from an interventional module × corruption sensitivity matrix (E5a).

## 4. Experiments and current results

| Exp | Question | Status | Headline (95% CI where available) | Evidence |
|---|---|---|---|---|
| E1 | Module performance and generalisation | done | M2 FF++ AUC 0.871 [0.849, 0.896] → Celeb-DF 0.769 [0.735, 0.806]; M3 MSU video 0.914 [0.853, 0.962]; M4 0.691 [0.586, 0.788]; M5 0.995 [0.985, 1.0]; M1 seen designs 1.000 but **unseen designs 0.548** | `experiments/results/tables/e1_e3.md`, `docs/MODULES.md` |
| E1b | Image-trained PAD on video | done | CelebA-Spoof-only M3 on MSU video: AUC 0.43, 89% of genuine FF++ videos rejected → adding video data: 0.914, 1.4% | `docs/MODULES.md` |
| E2 | Calibration | partial | ECE per module in E1; temperature scaling made M2's test ECE slightly worse (0.129 → 0.150): calibration shift between FF++ val and test | heads report |
| E3 | Fusion ablation (5 seeds) | done | F3 0.963±0.005 > F2 0.956 > F1 0.931 > F4 0.917 > F6 0.878 > F0 0.868 > F5 0.725. F3−F1 +0.036 [+0.026, +0.047]; **F4−F1 −0.005 [−0.020, +0.011] (n.s.)**; F6−F5 +0.180 [+0.154, +0.208] | `e1_e3.md` |
| E4 | Thresholds and veto | done | APCER target 1% (worst type) → test 5.1% (FaceShifter); genuine 55.8% accept / 42.2% review / 2.0% reject; veto not certifiable with 321 genuine samples | `docs/FUSION.md` |
| E5 | Corruption sensitivity and drift detection | running | see `experiments/results/e5/e5.md` once the Kaggle run finishes | `experiments/e5_corruptions.py` |
| E6 | Novel-attack detection (leave-one-method-out) | done (negative) | Held-out FaceShifter AUROC 0.39–0.57 in every space tried (CLIP+LBP, forgery vector, PCA/LDA); Celeb-DF fakes 0.70–0.92, **but Celeb-DF genuine (control) 0.76–0.86**, so this is domain, not forgery novelty | `docs/MONITORING.md`, build_reference report |
| E7 | Drift vs novelty (6 scenarios × 20 windows) | done | Proposed rule: macro accuracy 0.45, **false-novelty rate 0.00 in every scenario**. OOD-rate-only baseline: 0.40 with **95% false novelty under domain drift**. Rule without quality conditioning: 0.475 | build_reference report (`e7.py`) |
| E7b | iid vs subject-level null | done | stationary windows flagged as drift: 100% (iid) → 10% (subject-level null) | `docs/MONITORING.md` |
| E9 | Latency | measured | 8–37 s per session on laptop CPU (warm 8–15 s) | `docs/FUSION.md` |
| E10 | Bias / failure audit | not done | needs attribute labels (CelebA-Spoof attributes available) | — |

## 5. Contributions (reframed to what the evidence supports)
1. **C1, monitoring methodology (strongly supported):** naive iid two-sample and binomial OOD tests are invalid on biometric streams, because sessions are clustered by person. A subject-level empirical null cuts stationary false alarms from 100% to 10% at α = 0.05. *Experiment: E7b.*
2. **C2, drift-robust novelty triage (supported for the false-alarm side):** quality-conditional disambiguation never mislabels capture drift as a new attack (0% false novelty across all six scenarios). An OOD-rate-only monitor does so in 95% of domain-drift windows. *Limitation, reported prominently:* true novel attacks were not detected (0%), because a held-out manipulation method is not out-of-distribution in any representation we tried (E6). *Experiments: E6, E7.*
3. **C3, fusion (partially supported):** learned stacking of calibrated module scores with quality features significantly beats the fixed sum rule (+0.036 AUC, CI excludes 0). Quality-gated mixtures and Dempster–Shafer fusion do **not** help on clean composite data. *Experiment: E3*; E5 tests whether gating helps under degradation.
4. **C4, an honest evaluation protocol for multi-module KYC (resource contribution):**
   - composite sessions built only from held-out partitions;
   - leave-one-document-type-out, which exposes that document forensics does not transfer to new ID designs (0.548);
   - the image→video PAD gap;
   - certified veto thresholds;
   - all configs, Kaggle notebooks and hashes released.
   *Experiments: E1, E1b, E4.*
5. **C5, interventional recommendations (pending E5):** a module × corruption sensitivity matrix that turns drift alerts into concrete retraining actions.

## 6. Limitations (all go in the paper)
- Composite sessions make fraud signals independent across modules by construction.
- No real ID documents or KYC captures were used. M5's "ID photos" are video frames.
- The Kaggle mirrors are partial: CelebA-Spoof has 89.8% of the paper's images, the DF40 slice only 2 techniques, and the FF++ DFD folder is a subset.
- Small PAD/liveness sets (MSU-MFSD: 35 subjects). Challenge-response liveness is evaluated only qualitatively.
- M1 fails on unseen document designs. SIDTD has one forgery family.
- Novelty detection is a negative result in our representations. Quality features partly identify the source dataset.
- Single-site, offline evaluation; no human-review loop data, so performance drift (DDM/EDDM) is tested only synthetically.

## 7. Ethics
- Biometric data: only public research datasets under their original licences (Kaggle mirrors do not change those terms). No real ID documents.
- Privacy by design: encrypted media with retention TTL, no raw biometrics in logs, encrypted identity templates, erasure endpoint. GDPR Art. 9 and the DPDP Act are discussed in `docs/PRIVACY.md`.
- Dual use: the system describes detection only and releases no attack tooling.
- Demographic performance was not audited (E10 pending). This must be stated, and no fairness claims are made.

## 8. Candidate venues (no acceptance is promised)
- **IEEE Access:** fits a full-system paper with an honest negative result (C1–C4).
- **Pattern Recognition Letters:** a short paper focused on C1 + C2 (monitoring methodology).
- Stretch: **IJCB** or **IEEE T-BIOM**, if E5 supports C5 and a forgery-specific representation makes C2's detection side work.
- A workshop at a vision conference (e.g. a WACV workshop) for the drift-vs-novelty idea alone.

## 9. What would most strengthen the paper (next work)
1. A **forgery-attribution embedding** (supervised-contrastive over known methods) so that a held-out method becomes OOD, then re-run E6/E7.
2. Complete **E5** and test whether quality gating (F4/F6) helps under real corruptions.
3. **More unseen generators:** SadTalker/SimSwap from the DF40 slice, plus DFDC.
4. **E10 bias audit** on CelebA-Spoof attributes.
