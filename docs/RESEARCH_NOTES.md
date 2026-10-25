# Research Notes

Every entry below was checked during this session (2026-10-26) against an arXiv / CVF / publisher / official page reached through the link given. Venue and year come from that page. Where the conference venue could not be confirmed, only the arXiv ID is given. "Borrowed idea" is the single concrete thing this project takes from the work.

Module keys: M1 document, M2 deepfake, M3 replay/PAD, M4 liveness, M5 biometrics, Q quality vector, F fusion/calibration, D drift, N novelty, X explainability, R regulation/privacy.

---

## M1 — Document authenticity

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 1 | Krawetz, *A Picture's Worth: Digital Image Analysis and Forensics* | Black Hat USA 2007 (white paper, not peer-reviewed) | http://www.blackhat.com/presentations/bh-usa-07/Krawetz/Whitepaper/bh-usa-07-krawetz-WP.pdf | Error Level Analysis: re-save at known JPEG quality and inspect the residual. Used as the **weak baseline** only; it is known to be unreliable on its own. | M1 baseline features, X (ELA heatmap) |
| 2 | Kwon et al., *CAT-Net: Compression Artifact Tracing Network for Detection and Localization of Image Splicing* | WACV 2021 | https://openaccess.thecvf.com/content/WACV2021/papers/Kwon_CAT-Net_Compression_Artifact_Tracing_Network_for_Detection_and_Localization_of_WACV_2021_paper.pdf (journal extension: arXiv 2108.12947) | A separate DCT-coefficient stream: splices leave double-JPEG/quantization inconsistencies that RGB-only CNNs miss. We compute DCT-histogram periodicity / blockiness features. | M1 features |
| 3 | Chen et al., *Image Manipulation Detection by Multi-View Multi-Scale Supervision* (MVSS-Net) | ICCV 2021 | https://arxiv.org/abs/2104.06832 (extended MVSS-Net++: arXiv 2112.08935) | Noise-view (residual-filter) and boundary-artifact views; **specificity** on authentic images matters as much as sensitivity. We report false alarms on clean genuine IDs explicitly. | M1 features, evaluation protocol |
| 4 | Guillaro et al., *TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery Detection and Localization* | CVPR 2023 | https://arxiv.org/abs/2212.10957 · repo https://github.com/grip-unina/TruFor | Outputs a **reliability/confidence map** next to the anomaly map, so a detector says where its output can be trusted. This is the model for our per-module `confidence` output. Optional plug-in backend for M1. | M1 (optional deep backend), F (confidence) |
| 5 | Qu et al., *Towards Robust Tampered Text Detection in Document Image: New Dataset and New Solution* (DocTamper) | CVPR 2023 | https://openaccess.thecvf.com/content/CVPR2023/html/Qu_Towards_Robust_Tampered_Text_Detection_in_Document_Image_New_Dataset_CVPR_2023_paper.html | Text tampering in documents is visually consistent, so frequency/compression cues are needed. The dataset is **non-commercial and needs an institutional application**. | M1 data (optional), features |
| 6 | *MIDV-2020: A Comprehensive Benchmark Dataset for Identity Document Analysis* | arXiv 2021 | https://arxiv.org/abs/2107.00396 | 1000 **mock** identity documents with artificially generated faces. A real-person-free source of ID imagery. | M1 data, M5 (ID portrait) |
| 7 | *Synthetic dataset of ID and Travel Documents* (SIDTD) | Scientific Data 2024 | https://www.nature.com/articles/s41597-024-04160-9 · repo (CC-BY-4.0) https://github.com/Oriolrt/SIDTD_Dataset | Forged IDs derived from MIDV-2020 under a permissive license. **Primary M1 training/eval set.** | M1 data |
| 8 | *Document Liveness Challenge Dataset (DLC-2021)* | Journal of Imaging 8(7):181, 2022 | https://www.mdpi.com/2313-433X/8/7/181 | Synthetic IDs with **screen-recapture** attacks and a stated GDPR-compliant design. Recapture detection baseline. | M1 recapture features/data |
| 9 | Chen, Zhang, Lan, Huang, *Domain Generalization for Document Authentication against Practical Recapturing Attacks* | arXiv 2021 | https://arxiv.org/abs/2101.01404 | Reports document recapture in **APCER/BPCER at a fixed BPCER** across devices and document types. We adopt that reporting protocol for M1 recapture. | M1 evaluation |
| 10 | *FantasyID: A dataset for detecting digital manipulations of ID-documents* | arXiv 2025 | https://arxiv.org/abs/2507.20808 | Shows that TruFor and others fall to about 50% FNR at 10% FPR on realistic KYC ID forgeries. This is a **negative result** we cite to set honest expectations, and a candidate cross-dataset test set. | M1 evaluation, limitations |
| 11 | *IDNet: A Novel Dataset for Identity Document Analysis and Fraud Detection* | arXiv 2024 | https://arxiv.org/abs/2408.01690 | 837k synthetic ID images (~490 GB). A scale-up option if SIDTD is too small. | M1 data (optional) |
| 12 | ICAO Doc 9303, Part 3 (MRZ specifications, check digits) | ICAO standard | https://www.icao.int/sites/default/files/publications/DocSeries/9303_p3_cons_en.pdf | 7-3-1 weighted mod-10 check digits on MRZ fields. Gives a deterministic consistency check plus a cross-check of MRZ fields against the printed fields. | M1 rule features, synthetic ID generator |
| 13 | Scherhag et al., *Face Recognition Systems Under Morphing Attacks: A Survey* | IEEE Access 7, 2019 | https://dasec.h-da.de/wp-content/uploads/2019/02/morphdet_survey.pdf | Morphed ID portraits beat face matchers. Morphing is the threat model for the **held-out "novel attack"** scenario. | N (novel attack design), M5 |

## M2 — Video deepfake detection

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 14 | Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images* | ICCV 2019 | https://arxiv.org/abs/1901.08971 · access form via https://github.com/ondyari/FaceForensics | Standard benchmark with 4+ manipulation methods (DF, F2F, FS, NT, FaceShifter) at several compression levels. Used for **leave-one-method-out** and compression robustness. | M2 data, N (LOGO) |
| 15 | Li et al., *Celeb-DF: A Large-Scale Challenging Dataset for DeepFake Forensics* | CVPR 2020 | https://arxiv.org/abs/1909.12962 | High-quality face swaps; standard **cross-dataset** target after training on FF++. Several videos per identity let us build ID-portrait/selfie pairs. | M2 eval, M5 composite pairs |
| 16 | Dolhansky et al., *The DeepFake Detection Challenge (DFDC) Dataset* | arXiv 2020 | https://arxiv.org/abs/2006.07397 | Paid consenting actors, several generation methods, real-world perturbations. Optional cross-dataset test. | M2 eval (optional) |
| 17 | Yan et al., *DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection* | NeurIPS 2023 (Datasets & Benchmarks) | https://arxiv.org/abs/2307.01426 · https://github.com/SCLBD/DeepfakeBench | Unified preprocessing and protocols. We follow its frame sampling / face-crop conventions so numbers are comparable. | M2 protocol |
| 18 | Yan et al., *DF40: Toward Next-Generation Deepfake Detection* | NeurIPS 2024 (Datasets & Benchmarks) | https://arxiv.org/abs/2406.13495 | 40 distinct generation techniques (face-swap, reenactment, entire-face synthesis). The best source for **leave-one-generator-out** novelty experiments. | N (LOGO), M2 |
| 19 | Li et al., *Face X-ray for More General Face Forgery Detection* | CVPR 2020 | https://arxiv.org/abs/1912.13458 | Detect the **blending boundary** instead of generator-specific artifacts, so detection generalizes to unseen methods. | M2 design rationale |
| 20 | Shiohara & Yamasaki, *Detecting Deepfakes with Self-Blended Images* (SBI) | CVPR 2022 | https://arxiv.org/abs/2204.08376 · https://github.com/mapooon/SelfBlendedImages | Make pseudo-fakes by blending a real face with a perturbed copy of itself. **Training data from genuine videos only.** Lets us build an offline/sample pipeline without fake datasets. | M2 training data, offline demo |
| 21 | Ojha, Li, Lee, *Towards Universal Fake Image Detectors that Generalize Across Generative Models* (UnivFD) | CVPR 2023 | https://arxiv.org/abs/2302.10174 | A **frozen CLIP feature space + linear probe / nearest neighbour** generalizes across generators better than end-to-end real/fake training. This is our chosen M2 backbone, and it is CPU-feasible. | M2 backbone, N (embedding space) |
| 22 | Zheng et al., *Exploring Temporal Coherence for More General Video Face Forgery Detection* (FTCN) | ICCV 2021 | https://arxiv.org/abs/2108.06693 | Temporal incoherence is a generator-agnostic cue. We use lightweight temporal-consistency features (embedding and landmark jitter) instead of a full FTCN. | M2 temporal features |
| 23 | Frank et al., *Leveraging Frequency Analysis for Deep Fake Image Recognition* | ICML 2020 | https://arxiv.org/abs/2003.08685 | GAN upsampling leaves spectral artifacts. We use an azimuthally averaged spectrum as an M2 feature group. | M2 frequency features |
| 24 | Tan & Le, *EfficientNet* | ICML 2019 | https://arxiv.org/abs/1905.11946 | Backbone used by SBI. Optional GPU-trained M2 variant. | M2 (optional) |
| 25 | Ciftci, Demir, Yin, *FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals* | IEEE TPAMI 2020 | https://arxiv.org/abs/1901.02212 | Biological (PPG) signals are not preserved spatially or temporally in fakes. Our rPPG features are **shared between M4 and M2**. | M2, M4 |
| 26 | Li, Chang, Lyu, *In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking* | IEEE WIFS 2018 | https://arxiv.org/abs/1806.02877 | Abnormal blink statistics in early deepfakes. A weak cue on modern generators, but a cheap one. | M2/M4 blink features |

## M3 — Replay / presentation attack detection

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 27 | ISO/IEC 30107-3:2023, *Biometric presentation attack detection — Part 3: Testing and reporting* | ISO standard | https://www.iso.org/standard/79520.html | Terminology and metrics: **APCER, BPCER**, bona fide vs attack presentation, per-PAI-species reporting. | Evaluation, UI wording |
| 28 | Boulkenafet et al., *OULU-NPU: A Mobile Face Presentation Attack Database with Real-World Variations* | IEEE FG 2017 | https://oulurepo.oulu.fi/bitstream/handle/10024/24331/nbnfi-fe2019091328071.pdf | Protocols 1–4 test unseen environment, attack instrument and sensor. The **sensor-shift protocol** doubles as real-world covariate drift. | M3 data/protocol, D |
| 29 | Liu, Jourabloo, Liu, *Learning Deep Models for Face Anti-Spoofing: Binary or Auxiliary Supervision* (SiW) | CVPR 2018 | https://arxiv.org/abs/1803.11097 | Auxiliary depth and **rPPG** supervision generalizes better than binary labels. This justifies feeding rPPG evidence into both M3 and M4. | M3, M4 |
| 30 | Zhang et al., *CelebA-Spoof: Large-Scale Face Anti-Spoofing Dataset with Rich Annotations* | ECCV 2020 | https://arxiv.org/abs/2007.12342 | 625k images, 10 spoof types, illumination/environment attributes. Attributes enable **per-condition** analysis. | M3 data |
| 31 | Zhang et al., *A Dataset and Benchmark for Large-Scale Multi-Modal Face Anti-Spoofing* (CASIA-SURF) | CVPR 2019 | https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_A_Dataset_and_Benchmark_for_Large-Scale_Multi-Modal_Face_Anti-Spoofing_CVPR_2019_paper.html | RGB+depth+IR. **Not used**: browser webcams give RGB only. Cited to justify the RGB-only scope. | Scope/limitations |
| 32 | Yu et al., *Searching Central Difference Convolutional Networks for Face Anti-Spoofing* (CDCN) | CVPR 2020 | https://arxiv.org/abs/2003.04092 | Central-difference convolution captures fine gradient texture of spoof media. Optional GPU deep M3 backend. | M3 (optional) |
| 33 | Patel, Han, Jain, *Secure Face Unlock: Spoof Detection on Smartphones* | IEEE TIFS 11(10), 2016 | https://biometrics.cse.msu.edu/pubs/index.html | **Moiré aliasing**, surface reflection, colour distortion as replay cues. Our FFT-peak moiré features. | M3 features, M1 recapture |
| 34 | Wen, Han, Jain, *Face Spoof Detection with Image Distortion Analysis* | IEEE TIFS 2015 | http://biometrics.cse.msu.edu/Publications/Face/WenHanJain_FaceSpoofDetection_TIFS15.pdf | Hand-crafted IDA vector: specular reflection, blurriness, chromatic moments, colour diversity. A CPU-cheap M3 baseline. | M3 features |
| 35 | Boulkenafet, Komulainen, Hadid, *Face Spoofing Detection Using Colour Texture Analysis* | IEEE TIFS 2016 | https://www.researchgate.net/publication/301571761_Face_Spoofing_Detection_Using_Colour_Texture_Analysis | LBP texture on **chrominance** (HSV, YCbCr) channels, not only luminance. | M3 features |
| 36 | Anjos & Marcel, *Counter-Measures to Photo Attacks in Face Recognition: A Public Database and a Baseline* | IJCB 2011 | https://publications.idiap.ch/downloads/papers/2011/Anjos_IJCB_2011.pdf | **Correlation between head motion and scene/background motion**: a handheld photo or screen moves face and background together. | M3/M4 optical-flow feature |
| 37 | Liu et al., *Deep Tree Learning for Zero-Shot Face Anti-Spoofing* (SiW-M) | CVPR 2019 | https://arxiv.org/abs/1904.02860 | Treats **unknown spoof types** explicitly: routes samples to learned spoof sub-clusters. Prior art for open-set PAD. | N |

## M4 — Liveness (rPPG, blink, motion, challenge-response)

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 38 | de Haan & Jeanne, *Robust Pulse Rate from Chrominance-Based rPPG* (CHROM) | IEEE TBME 60, 2013 | https://pubmed.ncbi.nlm.nih.gov/23744659/ | Chrominance projection suppresses specular and motion components. | M4 rPPG |
| 39 | Wang, den Brinker, Stuijk, de Haan, *Algorithmic Principles of Remote PPG* (POS) | IEEE TBME 64(7), 2017 | https://pure.tue.nl/ws/files/31563684/TBME_00467_2016_R1_preprint.pdf | Plane-orthogonal-to-skin projection. Our **primary rPPG extractor**: training-free and CPU-cheap. | M4 rPPG |
| 40 | Liu et al., *rPPG-Toolbox: Deep Remote PPG Toolbox* | NeurIPS 2023 (Datasets & Benchmarks) | https://arxiv.org/abs/2210.00716 · https://github.com/ubicomplab/rPPG-Toolbox | Reference implementations of POS/CHROM and standard band-pass/FFT post-processing (0.75–2.5 Hz style). We validate our POS against it. | M4 implementation check |
| 41 | Li, Komulainen, Zhao, Yuen, Pietikäinen, *Generalized Face Anti-Spoofing by Detecting Pulse From Face Videos* | ICPR 2016 | https://oulurepo.oulu.fi/bitstream/handle/10024/25960/nbnfi-fe201902256226.pdf | Pulse-spectrum features (peak power and SNR in the HR band) separate live faces from masks and prints. | M4 rPPG features |
| 42 | Soukupová & Čech, *Real-Time Eye Blink Detection using Facial Landmarks* | CVWW 2016 | https://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf | **Eye Aspect Ratio (EAR)** time series from 6 eye landmarks. Blink challenge verification. | M4 blink |

## M5 — Facial geometry and face match

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 43 | Kartynnik et al., *Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs* | arXiv 2019 | https://arxiv.org/abs/1907.06724 | The 468-point MediaPipe Face Mesh topology. Landmark indices for geometric ratios. | M5, M4 |
| 44 | Grishchenko et al., *Attention Mesh: High-fidelity Face Mesh Prediction in Real-time* | arXiv 2020 | https://arxiv.org/abs/2006.10962 | Refined eye/lip landmarks used by the current MediaPipe Face Landmarker (478 points incl. iris). | M5, M4 |
| 45 | Ablavatski et al., *Real-time Pupil Tracking from Monocular Video for Digital Puppetry* | arXiv 2020 | https://arxiv.org/abs/2006.11341 | Iris landmarks. Iris diameter is a near-constant physical scale, so the IPD/iris ratio is scale-invariant. | M5 ratios |
| 46 | Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition* | CVPR 2019 | https://arxiv.org/abs/1801.07698 | Angular-margin embedding; cosine similarity for ID-to-selfie match. Pretrained InsightFace models are **non-commercial research only** (https://github.com/deepinsight/insightface, License section). | M5 face match |

## Q — Capture-quality vector

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 47 | Pech-Pacheco et al., *Diatom Autofocusing in Brightfield Microscopy: A Comparative Study* | ICPR 2000 | https://www.researchgate.net/publication/3887632_Diatom_autofocusing_in_brightfield_microscopy_A_comparative_study | **Variance of Laplacian** (and Tenengrad) as a sharpness measure. | Q `blur_laplacian_var` |
| 48 | Mittal, Moorthy, Bovik, *No-Reference Image Quality Assessment in the Spatial Domain* (BRISQUE) | IEEE TIP 2012 | https://live.ece.utexas.edu/publications/2012/TIP%20BRISQUE.pdf | No-reference, distortion-generic quality score from natural-scene statistics. Optional Q feature. | Q |

## F — Calibration and fusion

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 49 | Kittler, Hatef, Duin, Matas, *On Combining Classifiers* | IEEE TPAMI 20(3), 1998 | https://dl.acm.org/doi/10.1109/34.667881 | Theoretical framework; the **sum rule** is a strong baseline. This is our fixed-weight baseline F1. | F baseline |
| 50 | Fierrez-Aguilar et al., *Discriminative Multimodal Biometric Authentication Based on Quality Measures* | Pattern Recognition 2005 | https://biometrics.eps.uam.es/fierrez/files/2005_PR_DiscriminativeMultimodalQ_Fierrez.pdf | The fusion function **adapts per authentication attempt** to the estimated quality of each signal. This is the direct ancestor of our quality-gated fusion. | F quality gating |
| 51 | Poh et al., *Benchmarking Quality-Dependent and Cost-Sensitive Score-Level Multimodal Biometric Fusion Algorithms* | IEEE TIFS 2009 | https://dl.acm.org/doi/10.1109/TIFS.2009.2034885 | An evaluation design that gives fusion both scores and quality measures and tests under device-induced quality change. We copy this design for the gating ablation. | F evaluation |
| 52 | Alonso-Fernandez et al., *Quality-Based Conditional Processing in Multi-Biometrics: Application to Sensor Interoperability* | arXiv 2211.13554 (journal version, see link) | https://arxiv.org/abs/2211.13554 | Linear logistic-regression fusion so fused scores behave like log-likelihood ratios, conditioned on quality. This is our LR stacking (F2). | F stacking |
| 53 | Jacobs, Jordan, Nowlan, Hinton, *Adaptive Mixtures of Local Experts* | Neural Computation 3, 1991 | https://direct.mit.edu/neco/article-pdf/3/1/79/812104/neco.1991.3.1.79.pdf | A softmax **gating network** chooses expert weights per input. Our gate maps the quality vector to module weights. | F gating network |
| 54 | Guo, Pleiss, Sun, Weinberger, *On Calibration of Modern Neural Networks* | ICML 2017 | https://arxiv.org/abs/1706.04599 | Temperature scaling, plus ECE and reliability diagrams. | F calibration |
| 55 | Zadrozny & Elkan, *Transforming Classifier Scores into Accurate Multiclass Probability Estimates* | KDD 2002 | https://dl.acm.org/doi/10.1145/775047.775151 | Isotonic-regression calibration for non-NN scorers (GBM modules). | F calibration |
| 56 | Naeini, Cooper, Hauskrecht, *Obtaining Well Calibrated Probabilities Using Bayesian Binning* | AAAI 2015 | https://pmc.ncbi.nlm.nih.gov/articles/PMC4410090/ | ECE/MCE definitions. | F metrics |
| 57 | Sensoy, Kaplan, Kandemir, *Evidential Deep Learning to Quantify Classification Uncertainty* | NeurIPS 2018 | https://arxiv.org/abs/1806.01768 | Outputs as Dirichlet evidence, giving belief masses plus an explicit uncertainty mass `u`. | F evidential, module confidence |
| 58 | Han, Zhang, Fu, Zhou, *Trusted Multi-View Classification* | ICLR 2021 | https://arxiv.org/abs/2102.02051 (TPAMI extension: arXiv 2204.11423) | **Dempster–Shafer combination of per-view Dirichlet opinions.** A view with high uncertainty automatically counts for less. This is our uncertainty-aware fusion (F5). | F evidential fusion |

## D — Drift detection

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 59 | Gama, Žliobaitė, Bifet, Pechenizkiy, Bouchachia, *A Survey on Concept Drift Adaptation* | ACM Computing Surveys 46(4), 2014 | https://dl.acm.org/toc/csur/2014/46/4 | Taxonomy (covariate vs real/concept drift; abrupt/gradual/recurring) and evaluation of detection delay and false alarms. | D taxonomy, metrics |
| 60 | Gretton et al., *A Kernel Two-Sample Test* | JMLR 13, 2012 | https://jmlr.csail.mit.edu/papers/v13/gretton12a.html | MMD with permutation p-values for multivariate shift. | D multivariate |
| 61 | Lopez-Paz & Oquab, *Revisiting Classifier Two-Sample Tests* | ICLR 2017 | https://arxiv.org/abs/1610.06545 | C2ST: held-out accuracy of a reference-vs-current classifier, with a binomial null. Its feature importances localize the shift, and its probabilities give density ratios. | D multivariate, attribution, D/N rule |
| 62 | Rabanser, Günnemann, Lipton, *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift* | NeurIPS 2019 | https://arxiv.org/abs/1810.11953 | Best performer was **dimensionality reduction followed by univariate tests with multiple-testing correction**, and classifier tests are weak at small window sizes. This guides our choice of window sizes and tests. | D design |
| 63 | Lipton, Wang, Smola, *Detecting and Correcting for Label Shift with Black Box Predictors* (BBSE) | ICML 2018 | https://arxiv.org/abs/1802.03916 | Test shift on the **model's output distribution**. Separates label/prior shift (more attacks of a known type) from covariate shift. | D output drift, D/N rule |
| 64 | Bickel, Brückner, Scheffer, *Discriminative Learning Under Covariate Shift* | JMLR 10, 2009 | https://jmlr.csail.mit.edu/papers/v10/bickel09a.html | Estimate density ratios with a **discriminative classifier** instead of density estimation. Used to reweight the reference window by quality only. | D/N conditional test |
| 65 | Benjamini & Hochberg, *Controlling the False Discovery Rate* | JRSS-B 57, 1995 | https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x | BH FDR control across all per-feature tests. | D |
| 66 | Yurdakul & Naranjo, *Statistical Properties of the Population Stability Index* | Journal of Risk Model Validation 14(4), 2020 | https://www.risk.net/journal-of-risk-model-validation/7725371/statistical-properties-of-the-population-stability-index | PSI has a sampling distribution (approximately chi-square), so the "0.1/0.25" rules of thumb should come with sample-size-aware critical values. | D PSI thresholds |
| 67 | Bifet & Gavaldà, *Learning from Time-Changing Data with Adaptive Windowing* (ADWIN) | SIAM SDM 2007 | https://epubs.siam.org/doi/10.1137/1.9781611972771.42 | Adaptive window with change guarantees. Streaming baseline on score streams. | D streaming |
| 68 | Gama, Medas, Castillo, Rodrigues, *Learning with Drift Detection* (DDM) | SBIA 2004 | https://www.researchgate.net/publication/220974771_Learning_with_Drift_Detection | Error-rate control chart. Applied to the **human-review label stream** (performance drift). | D performance |
| 69 | Baena-García et al., *Early Drift Detection Method* (EDDM) | IWKDDS 2006 | https://www.researchgate.net/publication/245999704_Early_Drift_Detection_Method | Distance between errors, more sensitive to gradual drift. | D performance |
| 70 | Page, *Continuous Inspection Schemes* | Biometrika 41, 1954 | https://www.researchgate.net/publication/221936186_Continuous_Inspection_Schemes | CUSUM; the basis of Page–Hinkley. | D streaming |
| 71 | Montiel et al., *River: Machine Learning for Streaming Data in Python* | JMLR 22, 2021 | https://jmlr.org/papers/volume22/20-1380/20-1380.pdf | Reference implementations of ADWIN/PH/DDM/EDDM, used to **cross-check** our own implementations. | D baselines |

## N — Novel-attack (open-set / OOD) detection

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 72 | Lee, Lee, Lee, Shin, *A Simple Unified Framework for Detecting OOD Samples and Adversarial Attacks* | NeurIPS 2018 | https://arxiv.org/abs/1807.03888 | **Class-conditional Gaussians with a shared covariance**; Mahalanobis distance to the nearest known class. Our primary novelty score. | N |
| 73 | Liu, Wang, Owens, Li, *Energy-based Out-of-distribution Detection* | NeurIPS 2020 | https://arxiv.org/abs/2010.03759 | Energy score `-T·logsumexp(f/T)` on the classifier head. | N baseline |
| 74 | Liang, Li, Srikant, *Enhancing the Reliability of OOD Image Detection in Neural Networks* (ODIN) | ICLR 2018 | https://arxiv.org/abs/1706.02690 | Temperature scaling plus input perturbation. Needs input gradients, so it only applies to the M2 head. | N baseline |
| 75 | Sun, Ming, Zhu, Li, *Out-of-Distribution Detection with Deep Nearest Neighbors* | ICML 2022 | https://arxiv.org/abs/2204.06507 | k-th NN distance on **L2-normalized** embeddings, with no distributional assumption. | N |
| 76 | Ruff et al., *Deep One-Class Classification* (Deep SVDD) | ICML 2018 | https://github.com/lukasruff/Deep-SVDD | Learn a minimum-volume hypersphere around normal data. | N baseline |
| 77 | Liu, Ting, Zhou, *Isolation Forest* | IEEE ICDM 2008 | https://research.monash.edu/en/publications/isolation-forest/ | Isolation path length as an anomaly score. | N baseline |
| 78 | Breunig, Kriegel, Ng, Sander, *LOF: Identifying Density-Based Local Outliers* | ACM SIGMOD 2000 | https://dl.acm.org/doi/10.1145/335191.335388 | Local density ratio. | N baseline |
| 79 | Bendale & Boult, *Towards Open Set Deep Networks* (OpenMax) | CVPR 2016 | https://arxiv.org/abs/1511.06233 | Weibull (EVT) fit on distances to class mean activation vectors, plus an explicit "unknown" class. | N baseline |
| 80 | Yang et al., *OpenOOD: Benchmarking Generalized Out-of-Distribution Detection* | NeurIPS 2022 (Datasets & Benchmarks) | https://arxiv.org/abs/2210.07242 | Unified OOD evaluation (AUROC, FPR@95TPR, AUPR); reference implementations to check ours. | N evaluation |
| 81 | Bates, Candès, Lei, Romano, Sesia, *Testing for Outliers with Conformal p-values* | Annals of Statistics 2023 | https://arxiv.org/abs/2104.08279 | Turn any novelty score into a **valid p-value** by ranking against a held-out reference set. Under "no novelty" the share of p < α is about Binomial(n, α), which gives a principled window-level novelty test. | N, D/N rule |
| 82 | Campello, Moulavi, Sander, *Density-Based Clustering Based on Hierarchical Density Estimates* (HDBSCAN) | PAKDD 2013 | https://www.scribd.com/document/918339733/Density-Based-Clustering-Based-on-Hierarchical-Density-Estimates | Variable-density clustering with noise and cluster **stability/persistence**. Coherent new cluster means alert; noise points mean isolated outliers. | N clustering |
| 83 | McInnes, Healy, Astels, *hdbscan: Hierarchical Density Based Clustering* | JOSS 2(11), 2017 | http://www.theoj.org/joss-papers/joss.00205/10.21105.joss.00205.pdf | Implementation (also in scikit-learn ≥1.3 as `sklearn.cluster.HDBSCAN`). | N clustering |
| 84 | de Faria, Carvalho, Gama, *MINAS: Multiclass Learning Algorithm for Novelty Detection in Data Streams* | Data Mining and Knowledge Discovery 30, 2016 | https://link.springer.com/content/pdf/10.1007/s10618-015-0433-y.pdf | **Closest prior art for drift-vs-novelty**: buffer the "unknown" samples, cluster them, then decide whether a cohesive cluster *extends a known class* (drift) or is a *new class* (novelty). We extend this with quality-conditional testing and calibrated p-values. | D/N rule (core) |

## X — Explainability

| # | Paper | Venue / year | Link | Borrowed idea | Informs |
|---|---|---|---|---|---|
| 85 | Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* | ICCV 2017 | https://arxiv.org/abs/1610.02391 | Gradient-weighted class activation maps for the M1/M2 heatmaps. | X |

## R — Regulation and privacy (primary sources, not papers)

| # | Source | Link | What we take |
|---|---|---|---|
| 86 | Reserve Bank of India, Master Direction – KYC (V-CIP provisions; FAQ dated 2025-06-09) | https://www.rbi.org.in/commonman/Upload/English/FAQs/PDFs/KYC09062025.pdf | V-CIP is operator-assisted with liveness, geotagging, and recording/audit trail. Our system is an **assistive** scorer with human review, not a replacement for the official. |
| 87 | Digital Personal Data Protection Act, 2023 (Act 22 of 2023), Gazette of India | https://egazette.gov.in/WriteReadData/2023/247847.pdf | Consent, purpose limitation, erasure on withdrawal, data-fiduciary obligations. Drives the consent screen, retention TTL and deletion API. |

---

## Deliberately excluded

These are commonly cited in this area but were **not verified in this session**, so they are left out until checked: Wolpert's stacked generalization, FaceNet (Schroff et al.), the Farnebäck optical-flow paper, the Replay-Attack database paper (Chingovska et al.), DeLong's AUC test, and Ross & Jain "Information fusion in biometrics" (PRL 2003). Each will be verified before it is used in the paper.

## Licensing summary and Kaggle sources

Decision (milestone 2): **Kaggle is the primary data source.** Datasets are attached to Kaggle notebooks and are not downloaded to the laptop. **A Kaggle mirror does not change the original dataset's terms.** We follow the original license and cite the original paper. The machine-readable registry is `training/datasets.yaml`. "Unverified" means no specific mirror has been checked yet; see `docs/KAGGLE.md` §3 for the verification checklist.

| Logical name | Kaggle URL | Original dataset (paper) | Original license / access | Mirror note | Bundled in repo? |
|---|---|---|---|---|---|
| `ffpp_c23` | https://www.kaggle.com/datasets/xdxd003/ff-c23 (page found 2026-10-26) | FaceForensics++ [14] | FF++ Terms of Use (access form, research, no redistribution) | Unofficial mirror; original terms apply; contents still to be checked against the original | No |
| `dfdc` | https://www.kaggle.com/c/deepfake-detection-challenge | DFDC [16] | Competition data rules (accept on Kaggle) | Official host (~470 GB, mounted read-only) | No |
| `celebdf_v2` | unverified | Celeb-DF v2 [15] | Research-only terms (request form) | Must contain Celeb-real / Celeb-synthesis / YouTube-real + test list | No |
| `celeba_spoof` | unverified | CelebA-Spoof [30] | Non-commercial research agreement | Must keep `metas/` label + attribute files | No |
| `sidtd` | unverified (or self-upload, private) | SIDTD [7] | CC-BY-4.0 | Re-hosting allowed with attribution | No |
| `df40` | unverified | DF40 [18] | See DeepfakeBench/DF40 README | If unavailable, E6 uses FF++ leave-one-method-out | No |
| `doctamper` | institution-gated (later, via PES University) | DocTamper [5] | Non-commercial, institutional application | Not required by any milestone | No |
| `oulu_npu` | institution-gated (later) | OULU-NPU [28] | EULA (institutional) | Not required; E7-S6 uses OULU only if obtained | No |
| MIDV-2020 | not used directly (SIDTD is derived from it) | MIDV-2020 [6] | see dataset page (not verified) | — | No |
| DLC-2021 | not on Kaggle (unverified) | DLC-2021 [8] | see journal page | optional | No |

| Model asset | License | How obtained |
|---|---|---|
| InsightFace `buffalo_l` (ArcFace) [46] | Code MIT; **pretrained models non-commercial research only** | Kaggle dataset `codmav-weights-insightface` / runtime download. Behind the M5 interface; facenet-pytorch is a drop-in swap |
| facenet-pytorch VGGFace2 | see repo (not verified) | `codmav-weights-facenet` |
| open_clip ViT-B/16 (OpenAI weights) | see open_clip / OpenAI model card (not verified) | `codmav-weights-openclip-vitb16` |
| MediaPipe Face Landmarker | Apache-2.0 | `codmav-weights-mediapipe` |
| Vosk small English model | see alphacephei model page (not verified) | `codmav-weights-vosk` |
