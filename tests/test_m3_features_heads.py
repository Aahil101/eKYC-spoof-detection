"""Milestone 3 unit tests: extractors and heads on synthetic signals with known ground truth."""
import json
import pickle

import cv2
import numpy as np
import pytest

from ml.features import forensics as X
from ml.features import liveness as L
from ml.heads import IsotonicCal, TemperatureScaler, build_matrix, ece, fit_head, group_split, pad_metrics


# ---------------------------------------------------------------- rPPG
def test_pos_recovers_known_pulse():
    fs, secs, hr = 30.0, 10.0, 72.0
    t = np.arange(int(fs * secs)) / fs
    rng = np.random.default_rng(0)
    pulse = np.sin(2 * np.pi * hr / 60 * t)
    # skin-like baseline with pulse mostly in green (absorption) + illumination drift + noise
    rgb = np.stack([180 + 0.3 * pulse, 120 + 0.8 * pulse, 100 + 0.2 * pulse], 1)
    rgb *= (1 + 0.02 * np.sin(2 * np.pi * 0.1 * t))[:, None]
    rgb += rng.normal(0, 0.15, rgb.shape)
    s = L.bandpass(L.pos(rgb, fs), fs)
    st = L.spectrum_stats(s, fs)
    assert abs(st["hr_bpm"] - hr) < 3
    assert st["snr_db"] > 3
    noise = L.spectrum_stats(L.bandpass(L.pos(180 + rng.normal(0, 0.5, rgb.shape), fs), fs), fs)
    assert noise["snr_db"] < st["snr_db"] - 3          # flat replay-like signal scores much lower


def test_chrom_recovers_pulse():
    fs, t = 30.0, np.arange(300) / 30.0
    p = np.sin(2 * np.pi * 1.25 * t)
    rgb = np.stack([180 + 0.3 * p, 120 + 0.8 * p, 100 + 0.2 * p], 1)
    st = L.spectrum_stats(L.bandpass(L.chrom(rgb), fs), fs)
    assert abs(st["hr_bpm"] - 75) < 3


def test_blink_counter_hysteresis():
    t = np.arange(150) * 1000 / 30
    e = np.full(150, 0.30)
    for c in (30, 80, 120):                    # 3 blinks of ~4 frames, one noisy wobble at 60
        e[c:c + 4] = 0.12
    e[60] = 0.24                               # small dip: not a blink
    e[81] = 0.22                               # flicker inside a blink must not double-count
    assert L.count_blinks(e, t, 0.30) == 3


# ---------------------------------------------------------------- forensics
def test_mrz_icao_specimen():
    # ICAO Doc 9303 Part 4 specimen passport MRZ
    l1 = "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<"
    l2 = "L898902C36UTO7408122F1204159ZE184226B<<<<<10"
    assert X.mrz_check([l1, l2])["mrz_checks_pass_frac"] == 1.0
    bad = l2[:9] + "7" + l2[10:]               # wrong document-number check digit
    assert X.mrz_check([l1, bad])["mrz_checks_pass_frac"] == pytest.approx(2 / 3)
    assert X.mrz_check([])["mrz_present"] == 0.0


def test_moire_detects_screen_grid():
    rng = np.random.default_rng(0)
    base = cv2.GaussianBlur(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), (0, 0), 3)
    yy, xx = np.indices((256, 256))
    grid = (12 * np.sin(2 * np.pi * xx / 4.3) * np.sin(2 * np.pi * yy / 4.3))[..., None]
    screen = np.clip(base + grid, 0, 255).astype(np.uint8)
    assert X.moire_features(screen)["moire_peak_max"] > X.moire_features(base)["moire_peak_max"] + 1.0


def test_ela_and_noise_flag_spliced_region():
    rng = np.random.default_rng(1)
    img = cv2.GaussianBlur(rng.integers(0, 255, (256, 256, 3), dtype=np.uint8), (0, 0), 2)
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    doc = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    forged = doc.copy()
    forged[96:160, 96:160] = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)   # pasted noisy patch
    e0, _ = X.ela_features(doc)
    e1, _ = X.ela_features(forged)
    assert e1["ela_q90_block_maxz"] > e0["ela_q90_block_maxz"]
    n0, _ = X.noise_residual_features(doc)
    n1, _ = X.noise_residual_features(forged)
    assert n1["noise_block_maxz"] > n0["noise_block_maxz"]


def test_colour_lbp_shape_and_norm():
    img = np.random.default_rng(0).integers(0, 255, (128, 128, 3), dtype=np.uint8)
    h = X.colour_lbp(img)
    assert h.shape == (354,) and np.allclose(h.reshape(6, 59).sum(1), 1)


# ---------------------------------------------------------------- challenges
class _FF:
    def __init__(self, yaw=0.0, pitch=0.0, blend=None):
        self.yaw, self.pitch, self.blend, self.ok = yaw, pitch, blend or {}, True


def test_challenge_compliance_rules():
    from ml.common.interface import ChallengeStep
    t = list(np.arange(0, 7000, 66.7))
    n = len(t)
    yaw = np.zeros(n)
    yaw[(np.array(t) > 600) & (np.array(t) < 2500)] = 25           # turned left in step 1 window
    faces = [_FF(yaw=y, blend={"jawOpen": 0.3 if 2600 < tt < 4500 and int(tt / 200) % 2 else 0.02})
             for y, tt in zip(yaw, t)]
    ear = np.full(n, 0.3)
    for c in (75, 85):
        ear[c:c + 3] = 0.1                                           # two blinks in step 3 window
    steps = [ChallengeStep("turn_left", "", 500, 2500), ChallengeStep("read_digits", "", 2500, 4700, {"digits": [4, 7, 2]}),
             ChallengeStep("blink_twice", "", 4700, 7000)]
    f, res = L.challenge_compliance(steps, faces, t, ear, None, "none")
    assert [r["passed"] for r in res] == [True, True, True]
    assert res[1]["method"] == "mouth_motion_fallback"
    f2, res2 = L.challenge_compliance(steps, faces, t, ear, "four seven two", "vosk")
    assert res2[1]["passed"] and res2[1]["heard"][:3] == [4, 7, 2]
    f3, res3 = L.challenge_compliance(steps, faces, t, ear, "one two three", "vosk")
    assert not res3[1]["passed"] and f3["challenge_all_passed"] == 0.0
    still = [_FF() for _ in t]
    f4, res4 = L.challenge_compliance(steps[:1], still, t, ear, None, "none")
    assert not res4[0]["passed"]


# ---------------------------------------------------------------- heads / calibration
def _synthetic(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    recs = []
    for i in range(n):
        s = {"a": rng.normal(1.5 * y[i], 1), "b": rng.normal(-y[i], 1), "noise": rng.normal()}
        if rng.random() < 0.05:
            s["b"] = float("nan")                                   # missing values must be tolerated
        recs.append({"mx": s, "mx_vec": {"v": rng.normal(y[i] * 0.5, 1, 8).astype(np.float32)},
                     "group": f"g{i // 3}"})
    return recs, y


@pytest.mark.parametrize("kind,calib", [("logreg", "temperature"), ("hgb", "isotonic")])
def test_fit_head_metrics_and_calibration(kind, calib):
    recs, y = _synthetic()
    Xm, names, vd, cols = build_matrix(recs, "mx", ("v",))
    assert Xm.shape == (3000, 3 + 8) and cols[3] == "v[0]"
    split = group_split([r["group"] for r in recs], seed=0)
    assert set(split) == {"train", "cal", "val", "test"}
    groups_by_part = {p: {r["group"] for r, s in zip(recs, split) if s == p} for p in set(split)}
    assert not (groups_by_part["train"] & groups_by_part["test"])    # group-disjoint
    art = fit_head("MX", Xm, y, split, names, vd, ("v",), "mx", kind=kind, calib=calib)
    t = art.metrics["test"]
    assert t["auc"] > 0.9 and t["ece"] < 0.08
    assert 0.0 <= t["confidence_mean"] <= 1.0
    assert t["confidence_auroc_correct"] > 0.6                       # confidence predicts correctness
    p, z, c = art.predict(art.featurize(recs[0]))
    assert 0 < p[0] < 1 and 0 <= c[0] <= 1
    far = np.full((1, Xm.shape[1]), 50.0, np.float32)               # far out of distribution
    assert art.predict(far)[2][0] < 0.05                             # -> confidence collapses
    art2 = pickle.loads(pickle.dumps(art))
    assert np.allclose(art2.predict(Xm[:5])[0], art.predict(Xm[:5])[0])


def test_temperature_scaling_fixes_overconfidence():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    true_logit = (2 * y - 1) * 1.0 + rng.normal(0, 1, 5000)
    over = true_logit * 4                                            # overconfident model
    raw_ece = ece(1 / (1 + np.exp(-over)), y)
    ts = TemperatureScaler().fit(over, y)
    assert ts.T > 1.5                                                # shrinks overconfident logits
    assert ece(ts(over), y) < raw_ece / 2
    iso = IsotonicCal().fit(over, y)
    assert ece(iso(over), y) < raw_ece / 2


def test_pad_metrics_definitions():
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.65, 0.2])
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    m = pad_metrics(p, y, bpcer_target=0.25)
    assert m["auc"] == pytest.approx(15 / 16)
    # threshold = 25th percentile of genuine scores = 0.675 -> 1/4 genuine rejected, no attack accepted
    assert m["thr_at_bpcer"] == pytest.approx(0.675)
    assert m["bpcer"] == pytest.approx(0.25) and m["apcer"] == 0.0 and m["acer"] == pytest.approx(0.125)


# ---------------------------------------------------------------- M5 pairs
def test_m5_match_features_and_pairs():
    from ml.features.biometrics import RATIO_NAMES, match_features
    from training.feature_jobs import _m5_pairs
    rng = np.random.default_rng(0)
    centers = {f"id{i}": rng.normal(size=512) for i in range(6)}
    recs = []
    for ident, c in centers.items():
        for v in range(3):
            e = c + rng.normal(0, 0.3, (4, 512))
            e /= np.linalg.norm(e, axis=1, keepdims=True)
            recs.append({"label": 1, "identity": ident, "arc_embs": e,
                         "m5_ratios": {k: 0.5 for k in RATIO_NAMES}, "m5_geo": {"geo_frontal_frames": 10.0}})
    pairs = _m5_pairs(recs, 0)
    gen = [q["m5"]["arc_cos_mean"] for q, lab, _ in pairs if lab == 1]
    imp = [q["m5"]["arc_cos_mean"] for q, lab, _ in pairs if lab == 0]
    assert min(gen) > max(imp)
    f = match_features(None, np.zeros((0, 512)), None, {})
    assert np.isnan(f["arc_cos_mean"])


# ---------------------------------------------------------------- listers on fake trees
def test_listers(tmp_path):
    from training.feature_jobs import list_celebdf, list_ffpp, list_msu, list_sidtd
    ff = tmp_path / "ff" / "FaceForensics++_C23"
    for m in ("original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter"):
        (ff / m).mkdir(parents=True)
    for v in ("000", "003"):
        (ff / "original" / f"{v}.mp4").write_bytes(b"x")
    (ff / "Deepfakes" / "000_003.mp4").write_bytes(b"x")
    it = list_ffpp(tmp_path / "ff", {"000": "train", "003": "train"})
    assert [(i["label"], i["attack_type"], i["split"]) for i in it] == [(1, "bona_fide", "train")] * 2 + \
        [(0, "Deepfakes", "train")]

    cd = tmp_path / "cdf"
    for s in ("Celeb-real", "Celeb-synthesis", "YouTube-real"):
        (cd / s).mkdir(parents=True)
    (cd / "Celeb-real" / "id0_0000.mp4").write_bytes(b"x")
    (cd / "Celeb-real" / "id0_0001.mp4").write_bytes(b"x")
    (cd / "Celeb-synthesis" / "id0_id1_0000.mp4").write_bytes(b"x")
    (cd / "YouTube-real" / "00001.mp4").write_bytes(b"x")
    (cd / "List_of_testing_videos.txt").write_text("1 Celeb-real/id0_0001.mp4\n0 Celeb-synthesis/id0_id1_0000.mp4\n")
    it = {i["id"]: i for i in list_celebdf(cd, n_fake_extra=0)}
    assert it["celebdf/Celeb-real/id0_0001.mp4"]["split"] == "test"
    assert it["celebdf/Celeb-synthesis/id0_id1_0000.mp4"]["source_identity"] == "cdf1"
    assert it["celebdf/Celeb-real/id0_0000.mp4"]["identity"] == "cdf0"

    ms = tmp_path / "msu" / "x"
    (ms / "scene01" / "real").mkdir(parents=True)
    (ms / "scene01" / "attack").mkdir(parents=True)
    (ms / "train_sub_list.txt").write_text("02 03")
    (ms / "test_sub_list.txt").write_text("01")
    (ms / "scene01" / "real" / "real_client002_android_SD_scene01.mp4").write_bytes(b"x")
    (ms / "scene01" / "real" / "real_client002_android_SD_scene01.face").write_bytes(b"x")
    (ms / "scene01" / "attack" / "attack_client001_laptop_SD_printed_photo_scene01.mov").write_bytes(b"x")
    it = list_msu(tmp_path / "msu")
    assert len(it) == 2
    assert {(i["split"], i["attack_type"], i["label"]) for i in it} == {("train", "bona_fide", 1),
                                                                       ("test", "printed_photo", 0)}

    sd = tmp_path / "sidtd" / "templates" / "Images"
    (sd / "reals").mkdir(parents=True)
    (sd / "fakes").mkdir(parents=True)
    (sd / "reals" / "alb_id_83.jpg").write_bytes(b"x")
    (sd / "fakes" / "alb_id_83_fake_6_35.jpg").write_bytes(b"x")
    it = list_sidtd(tmp_path / "sidtd")
    assert {i["group"] for i in it} == {"alb_id_83"} and {i["label"] for i in it} == {0, 1}


def test_celeba_spoof_bbox_reference_frame():
    from ml.features.images import celeba_spoof_bbox
    img = np.zeros((599, 600, 3), np.uint8)
    assert celeba_spoof_bbox(img, "87 39 29 36 0.99") == (233, 104, 310, 200)   # verified visually on a real sample
    assert celeba_spoof_bbox(img, "garbage") is None


def test_train_heads_end_to_end(tmp_path, monkeypatch):
    """train_heads on synthetic shards for M1 (SIDTD-like) + M4 (MSU-like) -> artifacts + report."""
    from training.run import run
    import yaml
    rng = np.random.default_rng(0)
    feats = tmp_path / "out" / "feats"
    for ds, key in (("sidtd", "m1"), ("msu_mfsd", "m4")):
        recs = []
        for i in range(400):
            lab = int(i % 2)
            r = {"id": f"{ds}/{i}", "label": lab, "attack_type": "bona_fide" if lab else "x", "group": f"g{i // 4}",
                 "split": ("test" if i % 5 == 0 else "train") if ds == "msu_mfsd" else "pool",
                 key: {"f1": rng.normal(lab * 2), "f2": rng.normal()}}
            if key == "m1":
                r["m1_vec"] = {"spectrum": rng.normal(lab, 1, 24).astype(np.float32)}
            recs.append(r)
        (feats / ds).mkdir(parents=True)
        (feats / ds / "shard_00000.pkl").write_bytes(pickle.dumps(recs))
    cfg = yaml.safe_load(open("training/configs/train_heads.yaml"))
    cfg["params"]["modules"] = ["M1", "M4"]
    p = tmp_path / "th.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("CODMAV_FEATS_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("CODMAV_INPUT_ROOTS", "")
    m = run(p, "smoke", resume="none", out_root=tmp_path / "runs")
    assert m["status"] == "completed"
    assert set(m["artifacts"]) == {"head_M1.pkl", "head_M4.pkl", "heads_report.json"}
    rep = json.loads((tmp_path / "runs" / "train_heads" / "artifacts" / "heads_report.json").read_text())
    assert rep["modules"]["M1"]["metrics"]["test"]["auc"] > 0.9
