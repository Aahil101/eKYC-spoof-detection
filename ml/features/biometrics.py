"""M5 facial geometry (scale-invariant ratios + temporal stability) and ArcFace embeddings.

Ratios use MediaPipe landmarks [43-45]; iris diameter is a near-constant physical scale.
ArcFace (InsightFace w600k_r50, NON-COMMERCIAL research license) [46] runs via onnxruntime;
`FaceEmbedder` is the swap point for facenet-pytorch.
"""
from __future__ import annotations

import numpy as np

from ml.common import face as F

RATIO_NAMES = ("ipd_over_facewidth", "iris_over_ipd", "jaw_over_facelength", "eyenose_over_facelength",
               "nosemouth_over_facelength", "mouthwidth_over_ipd", "nosewidth_over_ipd", "face_aspect")


def _d(lm, a, b):
    return float(np.linalg.norm(lm[a, :2] - lm[b, :2]))


def ratios(lm: np.ndarray) -> dict[str, float]:
    ipd = _d(lm, F.LEFT_IRIS[0], F.RIGHT_IRIS[0])
    fw = _d(lm, F.JAW_LEFT, F.JAW_RIGHT)
    fl = _d(lm, F.FOREHEAD, F.CHIN)
    iris = 0.5 * (_d(lm, F.LEFT_IRIS[1], F.LEFT_IRIS[3]) + _d(lm, F.RIGHT_IRIS[1], F.RIGHT_IRIS[3]))
    eye_mid = 0.5 * (lm[F.LEFT_IRIS[0], :2] + lm[F.RIGHT_IRIS[0], :2])
    mouth_mid = 0.5 * (lm[F.MOUTH_LEFT, :2] + lm[F.MOUTH_RIGHT, :2])
    eps = 1e-6
    return {
        "ipd_over_facewidth": ipd / (fw + eps),
        "iris_over_ipd": iris / (ipd + eps),
        "jaw_over_facelength": _d(lm, F.JAW_LOW_LEFT, F.JAW_LOW_RIGHT) / (fl + eps),
        "eyenose_over_facelength": float(np.linalg.norm(eye_mid - lm[F.NOSE_TIP, :2])) / (fl + eps),
        "nosemouth_over_facelength": float(np.linalg.norm(lm[F.NOSE_TIP, :2] - mouth_mid)) / (fl + eps),
        "mouthwidth_over_ipd": _d(lm, F.MOUTH_LEFT, F.MOUTH_RIGHT) / (ipd + eps),
        "nosewidth_over_ipd": _d(lm, F.NOSE_LEFT, F.NOSE_RIGHT) / (ipd + eps),
        "face_aspect": fw / (fl + eps),
    }


def geometry_features(faces: list[F.FaceFrame], max_yaw: float = 15.0) -> tuple[dict[str, float], dict[str, float]]:
    """Median ratios over near-frontal frames + coefficient of variation (temporal stability)."""
    fr = [f for f in faces if f.ok and abs(f.yaw) <= max_yaw and abs(f.pitch) <= 20]
    nan = float("nan")
    if len(fr) < 3:
        return {f"geo_{k}": nan for k in RATIO_NAMES} | {f"geo_cv_{k}": nan for k in RATIO_NAMES} | \
            {"geo_frontal_frames": float(len(fr))}, {}
    R = {k: np.array([ratios(f.landmarks)[k] for f in fr]) for k in RATIO_NAMES}
    med = {k: float(np.median(v)) for k, v in R.items()}
    feats = {f"geo_{k}": med[k] for k in RATIO_NAMES}
    feats |= {f"geo_cv_{k}": float(np.std(v) / (abs(np.mean(v)) + 1e-9)) for k, v in R.items()}
    feats["geo_frontal_frames"] = float(len(fr))
    return feats, med


class FaceEmbedder:
    """ArcFace R50 via onnxruntime. Input: list of 112x112 BGR aligned crops. Output: L2-normalized (n,512)."""

    name = "arcface_w600k_r50"

    def __init__(self, model_path: str | None = None):
        import onnxruntime as ort

        from ml.common.weights import path_for
        p = str(model_path or path_for("arcface_buffalo_l"))
        prov = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in ort.get_available_providers()]
        so = ort.SessionOptions()
        so.log_severity_level = 3     # model declares batch=1 output shape; dynamic batches work but warn
        self.sess = ort.InferenceSession(p, sess_options=so, providers=prov)
        self.inp = self.sess.get_inputs()[0].name

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), np.float32)
        x = np.stack([c[:, :, ::-1] for c in crops]).astype(np.float32)   # BGR -> RGB
        x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
        e = self.sess.run(None, {self.inp: x})[0]
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def match_features(id_emb: np.ndarray | None, vid_embs: np.ndarray, id_ratios: dict[str, float] | None,
                   vid_ratios: dict[str, float]) -> dict[str, float]:
    nan = float("nan")
    f: dict[str, float] = {}
    if id_emb is None or len(vid_embs) == 0:
        f.update(arc_cos_mean=nan, arc_cos_max=nan, arc_cos_min=nan)
    else:
        c = vid_embs @ id_emb
        f.update(arc_cos_mean=float(c.mean()), arc_cos_max=float(c.max()), arc_cos_min=float(c.min()))
    if len(vid_embs) > 1:
        m = vid_embs @ vid_embs.T
        f["arc_selfconsistency"] = float(m[np.triu_indices(len(m), 1)].mean())
    else:
        f["arc_selfconsistency"] = nan
    for k in RATIO_NAMES:
        f[f"geo_delta_{k}"] = abs(vid_ratios[k] - id_ratios[k]) if (id_ratios and k in vid_ratios) else nan
    return f
