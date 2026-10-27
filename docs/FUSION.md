# Fusion and decision policy (milestone 4)

Code: `ml/fusion.py` (strategies, thresholds, veto, policy) and `training/fusion_job.py` (composite sessions + job). Kaggle run: `aahil1011/codmav-train-fusion` v4, on the heads from `codmav-train-heads` v3. Served bundle: **`kyc-v0.4` = 0.4.0+c229bacb6d** (five heads + `FUSION/fusion.pkl`). Single run, seed 0; bootstrap CIs follow in E1/E3.

## Composite sessions
No public dataset pairs an ID document with a selfie video, so sessions are assembled from **held-out** module data (EXPERIMENT_PLAN §0.3). Each part comes from the heads' calibration/validation partitions (→ `fusion_train`) or test partitions (→ `fusion_test`):

| type | document | video | ID portrait (M5) |
|---|---|---|---|
| bona_fide | real SIDTD | real FF++ / Celeb-DF / MSU-MFSD | other video of the same person (FF++: frame 0 of the same video) |
| doc_forgery | fake SIDTD | real | same person |
| deepfake_* | real | FF++ (5 methods) / Celeb-DF fake | the **impersonated source identity** |
| replay_* | real | MSU-MFSD attack (iPad, iPhone, printed photo) | the victim (same client) |

`fusion_train` has 1,430 sessions (375 genuine); `fusion_test` has 1,335 (303 genuine; attacks: 6 deepfake types × ~137, 97 doc forgery, 119 replay). Known leakage: the rows that calibrate the heads also fit the fusion. Fusion-test rows are never used for any choice.

## Strategy comparison (fusion_test)

| | strategy | AUC | EER | ECE | APCER @ BPCER 5% |
|---|---|---|---|---|---|
| F0 | mean of P | 0.872 | 0.185 | 0.042 | 0.632 |
| F1 | fixed weights ∝ AUC−0.5 (sum rule) | 0.932 | 0.150 | 0.040 | 0.302 |
| F2 | LR stacking [logit p, conf, missing] | 0.962 | 0.105 | 0.032 | 0.169 |
| **F3** | GBM stacking + quality vector (**selected**) | **0.968** | **0.076** | 0.059 | **0.103** |
| F4 | quality-gated mixture (MLP gate) | 0.927 | 0.153 | 0.036 | 0.309 |
| F5 | evidential Dempster–Shafer | 0.719 | 0.301 | 0.044 | 0.746 |
| F6 | DS + gated discounting | 0.898 | 0.189 | 0.054 | 0.299 |

The strategy was selected on an internal 70/30 split of `fusion_train` (F3 AUC 0.980 there), never on test.

**Findings (honest):**
- **Learned stacking (F2/F3) beats fixed fusion by a wide margin** (+0.03–0.10 AUC over F1/F0).
- **The quality-gated mixture (F4) does not beat fixed weights here** (0.927 vs 0.932). The synthetic unit test shows the gate does learn quality-dependent trust when such a dependence exists. On these sessions the capture quality varies little and is *dataset-identifying* (FF++ vs MSU), so the gate learns little that generalises. The corruption experiments (E5a/E3) are where this claim must be tested; on clean data it is **not supported**.
- **DS fusion (F5) performs poorly** because it depends on the module confidences. Several heads are confident but wrong on composite data (e.g. M1 on synthetic documents), and DS cannot down-weight confident errors. Quality-gated discounting (F6) recovers part of the loss (0.72 → 0.90).

**Leave-one-module-out (F3, fusion_test AUC):** without M5 0.833, without M1 0.914, without M2 0.958, without M3 0.966, without M4 0.971, all five 0.968. M5 and M1 carry the most information on this attack mix. Removing M4 *helps* slightly (liveness is weak and adds noise).

Single modules on the same sessions: M5 0.858, M2 0.773, M4 0.557, M1 0.547, M3 0.535. Each module only sees its own attack type, so these are low by construction.

## Decision policy (data-driven, never hand-set)
- **Thresholds** are chosen on *out-of-fold* fused scores over `fusion_train` (5 group folds). Choosing on in-sample scores gave train APCER 1% but **test 32%**, so out-of-fold scoring is required.
  - `T_high` = smallest t with **worst-case per-attack-type** acceptance ≤ 1%.
  - `T_low` = genuine rejection ≤ 2%.
- **Veto (hard rule):** only for modules whose AUC on their target attacks in `fusion_train` is ≥ 0.85 (M2 0.813 → not eligible; M3 0.998 → eligible). The veto is also **disabled unless the false-trigger target can be certified**: by the rule of three, ≤0.1% needs ≥3,000 genuine samples and only 321 exist. The M3 veto is therefore present but **disabled**, with the reason stored in the bundle.
- **Challenge rule:** a failed active-liveness challenge forces REJECT (retry), independent of the scores.

Result on `fusion_test` (targets: worst-type APCER ≤ 1%, BPCER ≤ 2%):

| genuine → ACCEPT | → REVIEW | → REJECT | attacks accepted (overall) | worst attack type accepted |
|---|---|---|---|---|
| 55.8% | 42.2% | 2.0% | 1.2% | **5.1% (FaceShifter)** |

The worst-type APCER target (1%) **does not hold on test** (5.1%). That is a calibration and generalisation gap on the hardest deepfake method, and it is reported as is.

Operating points (thresholds chosen on fusion_train per target, rates on fusion_test):

| APCER target | genuine auto-accept | genuine review | genuine reject | test worst-type APCER | test overall APCER |
|---|---|---|---|---|---|
| 0.5% | 0.0% | 98.0% | 2.0% | 0.0% | 0.0% |
| 1% | 55.8% | 42.2% | 2.0% | 5.1% | 1.2% |
| 2% | 57.4% | 40.6% | 2.0% | 5.1% | 1.2% |
| 5% | 74.3% | 23.8% | 2.0% | 8.0% | 2.2% |
| 10% | 80.9% | 17.2% | 2.0% | 11.7% | 3.1% |

## Live API check (`kyc-v0.4`, laptop CPU, FF++ clips + synthetic ID)
Both sessions end in **REJECT through the challenge rule**. FF++ clips are news footage, so nobody follows the random prompts; the rule works as designed. The fused score is ≈0 for both because **M1 = 0.00 on the synthetic test document** (out of the SIDTD distribution) and F3 weights M1 heavily. On a real webcam session with a SIDTD-like document this would differ. It also shows the deployment risk of a document module that has not been validated on new document designs (M1 leave-one-document-type-out AUC is 0.55, see MODULES.md).

## Limitations specific to fusion
- Composite sessions make fraud signals **independent across modules by construction**. Real attackers often combine attacks.
- Quality features partly identify the source dataset, which can inflate the stacking (F3) results. A dataset-balanced or dataset-blind variant belongs in E3.
- Replay attacks number only 119 in test (MSU-MFSD), and FaceShifter is the binding constraint for the APCER target.
